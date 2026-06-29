import os
import csv
from datetime import datetime
from app.api.schemas import FeedbackRequest

FEEDBACK_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "feedback.csv")

def save_feedback(feedback: FeedbackRequest) -> dict:
    """
    Appends the user feedback to a local CSV file.
    Creates the file and writes headers if it doesn't exist.
    """
    file_exists = os.path.isfile(FEEDBACK_FILE)
    
    with open(FEEDBACK_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write headers if it's a new file
        if not file_exists:
            writer.writerow(["Timestamp", "SessionID", "Score", "Question", "Answer"])
            
        # Write the feedback record
        writer.writerow([
            datetime.now().isoformat(),
            feedback.session_id,
            feedback.score,
            feedback.question,
            feedback.answer
        ])
        
    return {"status": "success", "message": "Feedback saved"}
