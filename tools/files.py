
import os
from tools.paths import WORKSPACE, get_path


def list_files():

    result = ""

    for folder in ["input", "knowledge", "output"]:

        folder_path = WORKSPACE / folder

        files = os.listdir(folder_path)

        result += folder + ": " + str(files) + "\n"

    return result


def read_file(filename):

    for folder in ["knowledge", "input"]:

        path = get_path(folder, filename)

        if path.is_file():

            if path.suffix.lower() not in [".txt", ".json", ".md"]:
                raise ValueError("Unsupported file type")

            with open(path, "r", encoding="utf-8") as file:

                return file.read()[:5000]

    return "File not found"
