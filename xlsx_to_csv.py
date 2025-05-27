import pandas as pd
import argparse
import sys


def convert_xlsx_to_csv(xlsx_file_path, csv_file_path, sheet_name=0):
    """
    Converts a sheet from an XLSX file to a CSV file.

    Args:
        xlsx_file_path (str): The path to the input XLSX file.
        csv_file_path (str): The path to the output CSV file.
        sheet_name (str or int, optional): The name or index of the sheet to convert.
                                           Defaults to 0 (the first sheet).
    """
    try:
        # Read the specified sheet from the Excel file
        df = pd.read_excel(xlsx_file_path, sheet_name=sheet_name)

        # Write the DataFrame to a CSV file
        # index=False prevents pandas from writing the DataFrame index as a column
        # encoding='utf-8' is a good default for broad compatibility
        df.to_csv(csv_file_path, index=False, encoding='utf-8')

        print(
            f"Successfully converted '{xlsx_file_path}' (sheet: {sheet_name if isinstance(sheet_name, str) else sheet_name+1}) to '{csv_file_path}'"
        )

    except FileNotFoundError:
        print(f"Error: The file '{xlsx_file_path}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert an XLSX file to a CSV file.")
    parser.add_argument("input_xlsx", help="Path to the input XLSX file.")
    parser.add_argument("output_csv", help="Path for the output CSV file.")
    parser.add_argument(
        "--sheet",
        default=0,
        help="Name or zero-based index of the sheet to convert (default: first sheet). Example: --sheet 'Sheet1' or --sheet 0",
    )

    args = parser.parse_args()

    # Determine if sheet is an integer (index) or string (name)
    sheet_to_convert = args.sheet
    try:
        sheet_to_convert = int(args.sheet)
    except ValueError:
        # If it's not an integer, treat it as a sheet name (string)
        pass

    convert_xlsx_to_csv(args.input_xlsx, args.output_csv, sheet_name=sheet_to_convert)
