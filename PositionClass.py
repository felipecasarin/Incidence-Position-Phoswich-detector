import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()


class PositionClass:
    def __init__(self, 
                 path: Optional[str] = None, 
                 base_folder_to_save: Optional[str] = None):
        self.path = path
        self.base_folder_to_save = base_folder_to_save or os.getenv("base_folder")

    def _check_file_exists(self, file_path):
        """Helper method to check if a file exists."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Error: The file '{file_path}' does not exist.")

    def _ensure_directory_exists(self, save_path):
        """Helper method to ensure directories exist before saving."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)


