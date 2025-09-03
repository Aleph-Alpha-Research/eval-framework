import atexit
import logging
import random
import re
import signal
import sqlite3
import threading
from enum import Enum
from time import sleep
from typing import Any
from uuid import uuid4

import docker
import mysql.connector
import mysql.connector.abstracts
import psycopg2  # type: ignore
from pydantic import BaseModel

from eval_framework.llm.base import BaseLLM
from eval_framework.metrics.base import MetricResult
from eval_framework.metrics.llm.base import BaseLLMJudgeMetric
from eval_framework.metrics.llm.graders.language import Language
from eval_framework.metrics.llm.graders.sql_quality_grader import SqlQualityGrader
from eval_framework.shared.types import Completion, LanguageMetricContext, extract_context_metric
from eval_framework.tasks.utils import get_docker_address

logger = logging.getLogger(__name__)


class SqlDialects(Enum):
    sqlite = "sqlite"
    postgres = "postgresql"
    mysql = "mysql"
    standard_sql = "standard_sql"


class SqlOutputComparison(BaseModel):
    matches_results_count: bool
    matches_column_count: bool
    results_equal: bool


class SqlValidationResult(BaseModel):
    success: bool
    schema_error: str | None = None
    query_error: str | None = None
    results: list[Any] = []


class LLMJudgeSqlMetricContext(LanguageMetricContext):
    dialect: str
    db_schema: str


_DOCKER_LAUNCH_LOCK = threading.Lock()
_MYSQL_PORT = 0
_POSTGRES_PORT = 0


class LLMJudgeSql(BaseLLMJudgeMetric):
    NAME = "SQL Quality"

    def __init__(self, llm_judge: BaseLLM):
        super().__init__(llm_judge)
        self._grader = SqlQualityGrader(llm_judge)

        self.postgres_password = "mysecretpassword"
        self.postgres_user = "postgres"

        self.mysql_password = "mysecretpassword"
        self.mysql_user = "root"
        self.mysql_db_name = "mysql"

        with _DOCKER_LAUNCH_LOCK:
            if _MYSQL_PORT != 0 and _POSTGRES_PORT != 0:
                return
            self.client = docker.from_env()
            atexit.register(self._shutdown_dbs)
            signal.signal(signal.SIGTERM, lambda *_: self._shutdown_dbs())
            self._start_postgres_db()
            self._start_mysql_db()
            self._wait_for_db_containers()

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [
                MetricResult(metric_name=f"{self.NAME}/{k}", value=None, higher_is_better=True, error=response.error)
                for k in [
                    "successfully_runs",
                    "is_just_sql",
                    "matches_results_count",
                    "matches_column_count",
                    "results_equal",
                    "llm_quality_score",
                ]
            ]

        context = extract_context_metric(response, LLMJudgeSqlMetricContext)

        assert isinstance(response.ground_truth, str)

        schema_id = str(uuid4()).replace("-", "_")

        expected_result = self.validate_query(
            SqlDialects(context.dialect),
            context.db_schema,
            response.ground_truth,
            f"golden_{schema_id}",
        )
        completion_stripped = response.completion.strip().strip("```sql").strip("```")
        completion_query = extract_query_from_completions(completion_stripped)
        if completion_query:
            result = self.validate_query(
                SqlDialects(context.dialect),
                context.db_schema,
                completion_query,
                f"completion_{schema_id}",
            )
        else:
            result = None

        results = [
            MetricResult(
                metric_name=f"{self.NAME}/successfully_runs",
                value=float(result is not None and result.success),
                higher_is_better=True,
                error=response.error,
            ),
            MetricResult(
                metric_name=f"{self.NAME}/is_just_sql",
                value=float(completion_query == completion_stripped),
                higher_is_better=True,
                error=response.error,
            ),
        ]

        if result is not None and result.success:
            output_comparison = SqlOutputComparison(
                matches_results_count=len(expected_result.results) == len(result.results),
                matches_column_count=count_result_columns(expected_result.results)
                == count_result_columns(result.results),
                results_equal=expected_result.results == result.results,
            )
            results.extend(
                [
                    MetricResult(
                        metric_name=f"{self.NAME}/matches_results_count",
                        value=float(output_comparison.matches_results_count),
                        higher_is_better=True,
                        error=response.error,
                    ),
                    MetricResult(
                        metric_name=f"{self.NAME}/matches_column_count",
                        value=float(output_comparison.matches_column_count),
                        higher_is_better=True,
                        error=response.error,
                    ),
                    MetricResult(
                        metric_name=f"{self.NAME}/results_equal",
                        value=float(output_comparison.results_equal),
                        higher_is_better=True,
                        error=response.error,
                    ),
                ]
            )

        grading = self._grader.grade(
            prompt=response.user_instruction,
            completion=completion_stripped,
            result=result.results if result and result.success else None,
            language=Language(response.get_instruction_language()),
        )
        results.append(
            MetricResult(
                metric_name=f"{self.NAME}/llm_quality_score",
                # [0, 1] normalization required for visualizer
                value=(float(grading.query_quality) - 1) / 4 if grading.query_quality is not None else None,
                higher_is_better=True,
                llm_judge_prompt=grading.judge_prompt,
                llm_judge_response=grading.judge_response,
                error=response.error,
            )
        )

        return results

    def _start_postgres_db(self) -> None:
        global _POSTGRES_PORT
        for _ in range(10):  # find a free port
            try:
                _POSTGRES_PORT = random.randint(1000, 65535)
                self.postgres_docker = self.client.containers.run(
                    "docker.io/postgres",
                    environment={"POSTGRES_PASSWORD": self.postgres_password},
                    ports={5432: _POSTGRES_PORT},
                    tty=True,
                    auto_remove=True,
                    detach=True,
                    network_mode="bridge",
                )
                break
            except docker.errors.APIError as e:
                if "port is already allocated" not in str(e):
                    raise e
                continue

    def _start_mysql_db(self) -> None:
        global _MYSQL_PORT
        for _ in range(10):  # find a free port
            try:
                _MYSQL_PORT = random.randint(1000, 65535)
                self.mysql_docker = self.client.containers.run(
                    "docker.io/mysql:latest",
                    environment={"MYSQL_ROOT_PASSWORD": self.mysql_password, "MYSQL_DATABASE": self.mysql_db_name},
                    ports={3306: _MYSQL_PORT},
                    tty=True,
                    auto_remove=True,
                    detach=True,
                    network_mode="bridge",
                )
                break
            except docker.errors.APIError as e:
                if "port is already allocated" not in str(e):
                    raise e
                continue

    def _wait_for_db_containers(self) -> None:
        for _ in range(600):
            try:
                con = self.connect_to_postgres()
                con.close()
                con = self.connect_to_mysql()
                con.close()
                return
            except Exception:
                logger.info("Could not connect to DBs yet...")
                sleep(1)
        raise Exception("DBs not available.")

    def _shutdown_dbs(self) -> None:
        if hasattr(self, "postgres_docker"):
            self.postgres_docker.kill()
        if hasattr(self, "mysql_docker"):
            self.mysql_docker.kill()

    def validate_query(
        self,
        dialect: SqlDialects,
        create_db_statements: str,
        sql_query: str,
        db_schema: str,
    ) -> SqlValidationResult:
        match dialect:
            case SqlDialects.sqlite | SqlDialects.standard_sql:
                return self.validate_query_sqlite(create_db_statements, sql_query, f"{dialect.value}_{db_schema}")
            case SqlDialects.postgres:
                return self.validate_query_postgres(create_db_statements, sql_query, f"{dialect.value}_{db_schema}")
            case SqlDialects.mysql:
                return self.validate_query_mysql(create_db_statements, sql_query, f"{dialect.value}_{db_schema}")
            case _:
                raise NotImplementedError(f"Query validation not implemented for {dialect.value}.")

    def validate_query_sqlite(self, create_db_statements: str, sql_query: str, db_schema: str) -> SqlValidationResult:
        con = sqlite3.connect(":memory:")
        cur = con.cursor()
        try:
            statements = separate_statements(create_db_statements)
            for statement in statements:
                cur.execute(statement)
                con.commit()
        except Exception as e:
            logger.info(f"Create statements are not compatible with SQLite. Reason: {e}")
            return SqlValidationResult(success=False, schema_error=str(e))
        try:
            queries = separate_statements(sql_query)
            for query in queries:
                cur.execute(query)
                con.commit()
            results = cur.fetchall()
        except Exception as e:
            logger.info(f"SQL query is not compatible with SQLite. Reason: {e}")
            return SqlValidationResult(success=False, query_error=str(e))

        con.close()
        return SqlValidationResult(success=True, results=results)

    def connect_to_postgres(self) -> psycopg2.extensions.connection:
        conn_params = {
            "dbname": "postgres",
            "user": self.postgres_user,
            "password": self.postgres_password,
            "host": get_docker_address(),
            "port": _POSTGRES_PORT,
        }
        return psycopg2.connect(**conn_params)

    def validate_query_postgres(self, create_db_statements: str, sql_query: str, db_schema: str) -> SqlValidationResult:
        con = self.connect_to_postgres()
        cur = con.cursor()
        cur.execute(f"CREATE SCHEMA {db_schema};")
        con.commit()
        cur.execute(f"ALTER USER {self.postgres_user} set SEARCH_PATH = {db_schema};")
        con.commit()
        try:
            statements = separate_statements(create_db_statements)
            for statement in statements:
                cur.execute(statement)
            con.commit()
        except Exception as e:
            logger.info(f"Create statements are not compatible with PostgreSQL. Reason: {e}")
            return SqlValidationResult(success=False, schema_error=str(e))
        try:
            queries = separate_statements(sql_query)
            for query in queries:
                cur.execute(query)
                con.commit()
            results = cur.fetchall()
        except Exception as e:
            logger.info(f"SQL query is not compatible with PostgreSQL. Reason: {e}")
            return SqlValidationResult(success=False, query_error=str(e))

        con.commit()

        con.close()
        return SqlValidationResult(success=True, results=results)

    def connect_to_mysql(
        self,
    ) -> mysql.connector.pooling.PooledMySQLConnection | mysql.connector.abstracts.MySQLConnectionAbstract:
        conn_params = {
            "database": self.mysql_db_name,
            "user": self.mysql_user,
            "password": self.mysql_password,
            "host": get_docker_address(),
            "port": _MYSQL_PORT,
        }
        return mysql.connector.connect(**conn_params)

    def validate_query_mysql(self, create_db_statements: str, sql_query: str, db_schema: str) -> SqlValidationResult:
        con = self.connect_to_mysql()
        cur = con.cursor(buffered=True)
        cur.execute(f"CREATE SCHEMA {db_schema};")
        con.commit()
        cur.execute(f"USE {db_schema};")
        try:
            statements = separate_statements(create_db_statements)
            for statement in statements:
                cur.execute(statement)
                con.commit()
        except Exception as e:
            logger.info(f"Create statements are not compatible with MySQL. Reason: {e}")
            con.close()
            return SqlValidationResult(success=False, schema_error=str(e))
        try:
            queries = separate_statements(sql_query)
            for query in queries:
                cur.execute(query)
                con.commit()
            results = cur.fetchall()
        except Exception as e:
            logger.info(f"SQL query is not compatible with MySQL. Reason: {e}")
            con.close()
            return SqlValidationResult(success=False, query_error=str(e))

        cur.close()
        con.close()
        return SqlValidationResult(success=True, results=results)


def separate_statements(statements: str) -> list[str]:
    return statements.split(";")[:-1]


def is_create_table_statement(statement: str) -> bool:
    return "CREATE TABLE" in statement


def count_result_columns(result: list[Any]) -> int:
    if len(result) == 0:
        return 0
    return len(result[0])


def extract_query_from_completions(completion: str) -> str | None:
    # Match SQL blocks starting with SELECT or WITH at line start
    # (allowing punctuation/whitespace), ending at first semicolon
    pattern = re.compile(r"(?:^|\n)[^a-zA-Z0-9_]*((?:select|with)\b.*?;)", re.IGNORECASE | re.DOTALL)

    matches = pattern.findall(completion)

    # Return the query only if exactly one match is found
    if len(matches) == 1:
        return matches[0].strip()
    return None
