import pandas as pd

def excel_to_csv_specific_columns(excel_file_path, csv_file_path, columns_to_keep):
    """
    Converts an Excel file to a CSV file, keeping only specified columns.

    Args:
        excel_file_path (str): The path to the input Excel file.
        csv_file_path (str): The path where the output CSV file will be saved.
        columns_to_keep (list): A list of column names to extract from the Excel file.
    """
    try:
        # Read the Excel file
        df = pd.read_excel(excel_file_path)

        # Select only the desired columns
        df_selected = df[columns_to_keep]

        # Save the selected columns to a CSV file
        df_selected.to_csv(csv_file_path, index=False, encoding='utf-8')

        print(f"Successfully converted '{excel_file_path}' to '{csv_file_path}' "
              f"with columns: {columns_to_keep}")

    except FileNotFoundError:
        print(f"Error: The file '{excel_file_path}' was not found.")
    except KeyError as e:
        print(f"Error: One of the specified columns was not found in the Excel file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # --- Configuration ---
    excel_input_file = "your_excel_file.xlsx"  # Replace with your Excel file name
    csv_output_file = "output_data.csv"      # Desired name for your output CSV file
    desired_columns = ["Summary_conclusion", "Current_disease"] # Columns to extract

    # --- Run the conversion ---
    excel_to_csv_specific_columns(excel_input_file, csv_output_file, desired_columns)