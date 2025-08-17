import numpy as np
import pytest

from eval_framework.tasks.utils import Editor, HatPaperEditor


class TestEditor:
    @pytest.fixture
    def editor(self) -> Editor:
        return Editor()

    def test_split_recombine(self, editor: Editor) -> None:
        sentence = "Here's an example sentence.\nAnd ... Another one!"
        assert editor._recombine(*editor._split_sentence(sentence)) == sentence

        sentence = "The quick brown fox\nIt jumped"
        assert editor._recombine(*editor._split_sentence(sentence)) == sentence

        sentence = "\nLeading and trailing\n\n"
        assert editor._recombine(*editor._split_sentence(sentence)) == sentence

        sentence = "...!"
        assert editor._recombine(*editor._split_sentence(sentence)) == sentence

        sentence = "word"
        assert editor._recombine(*editor._split_sentence(sentence)) == sentence

        sentence = " leading"
        assert editor._recombine(*editor._split_sentence(sentence)) == sentence

        sentence = "trailing "
        assert editor._recombine(*editor._split_sentence(sentence)) == sentence

    def test_get_word_probs(self, editor: Editor) -> None:
        words = ["The", "quick", "fox"]
        probs = editor._get_word_probs(words)
        np.testing.assert_equal(probs, np.array([0.25, 0.5, 0.25]))

    def test_transpose(self, editor: Editor) -> None:
        assert editor._transpose("The", 1, 2) == "Teh", "Editor does not correctly transpose letters"

        assert editor._transpose("brown", 2, 1) == "borwn", "Editor does not correctly transpose letters"

        with pytest.raises(AssertionError):
            editor._transpose("The", 0, 2)

    def test_delete(self, editor: Editor) -> None:
        assert editor._delete("The", 1) == "Te", "Editor does not correctly delete letters"

    def test_insert(self, editor: Editor) -> None:
        assert editor._insert("The", 1, "a") == "Tahe", "Editor does not correctly insert letters"

        with pytest.raises(AssertionError):
            editor._insert("The", 0, "ab")

    def test_change_casing(self, editor: Editor) -> None:
        assert editor._change_casing("The", 1) == "THe", "Editor does not correctly split words"
        assert editor._change_casing("THE", 1) == "ThE", "Editor does not correctly split words"
        assert editor._change_casing("T99", 1) == "T99", "Editor does not correctly split words"

    def test_split_word(self, editor: Editor) -> None:
        assert editor._split_word("The", 1) == "T he", "Editor does not correctly split words"

    def test_call(self, editor: Editor) -> None:
        assert editor("The quick brown fox.", 0.0) == "The quick brown fox."
        assert editor("The quick brown fox.", 0.1) == "The qick brown fox."
        assert editor("The quick brown fox.", 0.25) == "Thie qUick Bron fox."
        assert editor("The quick brown fox.", 0.5) == "The q uick cBrow n f   x."
        assert editor("The quick brown fox.", 1.0) == "TwH e mql UCik vbro CWn fox."
        assert editor("The quick brown fox.", 1.0, ["QuicK", "fox."]) == "dhET quick brOn fOx."

        editor2 = Editor(seed=22)
        assert editor2("The quick brown fox.", 0.0) == "The quick brown fox."
        assert editor2("The quick brown fox.", 0.1) == "The quick bRown fox."


class TestHatPaperEditor:
    @pytest.fixture
    def editor(self) -> HatPaperEditor:
        return HatPaperEditor()

    def test_permute_chars_in_string(self, editor: HatPaperEditor) -> None:
        result = editor.permute_chars_in_string("The quick brown fox.", 1.0, ["quicK"])
        assert result == "Teh quick bwrno fxo."
        result = editor.permute_chars_in_string("The quick brown fox.", 0.2)
        assert result == "The quikc brown fox."

    def test_replace_chars_in_string(self, editor: HatPaperEditor) -> None:
        result = editor.replace_chars_in_string("The quick brown fox.", 1.0, ["tHE"])
        assert result == "The qw%,k bWl.n ffx."
        result = editor.replace_chars_in_string("The quick brown fox.", 0.2)
        assert result == "T,e quick brown fox."

    def test_delete_chars_in_string(self, editor: HatPaperEditor) -> None:
        result = editor.delete_chars_in_string("The quick brown fox.", 1.0, ["fox"])
        assert result == "Te qk bn fox."
        result = editor.delete_chars_in_string("The quick brown fox.", 0.2)
        assert result == "The qick brown fox."

    def test_upper_case_string(self, editor: HatPaperEditor) -> None:
        result = editor.upper_case_string("The quick brown fox.")
        assert result == "THE QUICK BROWN FOX."
