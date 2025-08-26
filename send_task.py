# send_task.py
import time
from app import process_csv_file
import os # We'll need this for checking file existence

if __name__ == '__main__':
    # --- IMPORTANT CHANGE HERE ---
    # The file path is now corrected to match your directory structure from the screenshot.
    file_path = 'sales.csv' 

    try:
        # Check if the file exists at the specified path before trying to open it
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file '{file_path}' was not found. Please ensure it exists.")
        
        # Read the file content and pass it directly to the task
        with open(file_path, 'r') as f:
            file_content = f.read()

        # Define the column mapping as before. You might need to adjust this
        # based on the columns in your actual 'sales.csv' file.
        column_mapping = {
            'date_col': 'Date',
            'item_count_col': 'Users', # Or whatever column name corresponds to item count
            'revenue_col': 'Pageviews', # Or whatever column name corresponds to revenue
            'category_col': 'Category' # Add a category column if it exists in sales.csv
        }

        print(f"Reading file: {file_path} and sending its content to Celery...")
        
        # Call the task with the file content instead of the path
        # Also, we now pass the file content directly, not just the path.
        task_result = process_csv_file.delay(file_content, column_mapping)
        
        print(f"Task ID: {task_result.id}")
        print("The task has been queued. Waiting for it to complete...")
        
        while not task_result.ready():
            print(f"Current status: {task_result.status}. Waiting...")
            time.sleep(1)
            
        print(f"Task completed with status: {task_result.status}")
        
        if task_result.successful():
            print("Task was successful!")
            print(f"Result (if any): {task_result.result}")
        else:
            print("Task failed!")
            print(f"Failure reason: {task_result.info}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

