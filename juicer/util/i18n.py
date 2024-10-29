from gettext import translation, gettext as original_gettext
import os
from typing import Callable

# Define default locale directory and language
LOCALE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "i18n", "locales"
)
DEFAULT_LANGUAGE = "pt"

translator: Callable[..., str] = None


def set_language(language_code: str = DEFAULT_LANGUAGE) -> None:
    global translator
    try:
        translator = translation(
            "messages",
            localedir=LOCALE_DIR,
            languages=[language_code],
            fallback=True,
        )
    except Exception as e:
        print(f"Error setting language to {language_code}: {e}")
        translator = original_gettext


def gettext(message):
    return translator.gettext(message)


# Initialize the default language
set_language()
