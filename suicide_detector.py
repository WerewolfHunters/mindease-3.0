import smtplib
import os
from email.message import EmailMessage
from dotenv import load_dotenv


class MentalHealthMonitor:
    def __init__(self, sender_email, sender_password, email_template_file='format.txt'):
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.email_template_file = email_template_file

    def _read_email_template(self):
        with open(self.email_template_file, 'r') as file:
            content = file.read()
        return content

    def _send_email(self, recipient_email, email_content):
        print(f"[email] Preparing SMTP send. sender={self.sender_email}, recipient={recipient_email}")
        msg = EmailMessage()
        
        # Split subject and body from the content
        lines = email_content.strip().split('\n')
        subject_line = lines[0].replace("Subject:", "").strip()
        body = '\n'.join(lines[1:]).strip()

        msg['Subject'] = subject_line
        msg['From'] = self.sender_email
        msg['To'] = recipient_email
        msg.set_content(body)

        try:
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                print("[email] SMTP SSL connection established.")
                smtp.login(self.sender_email, self.sender_password)
                print("[email] SMTP login successful.")
                smtp.send_message(msg)
                print("[email] SMTP send_message successful.")
                return True
        except Exception as e:
            print(f"[email] SMTP send failed: {e}")
            return False

    def evaluate_and_notify(self, label_counts: dict, user_email: str):
        total = sum(label_counts.values())
        suicide_count = label_counts.get('suicide', 0)
        
        if total == 0:
            print("No labels to evaluate.")
            return False

        percentage = (suicide_count / total) * 100
        print(f"Suicide percentage: {percentage:.2f}%")

        if percentage >= 3:
            print("Threshold exceeded. Sending alert email...")
            if not self.sender_email or not self.sender_password:
                print("[email] Sender email/password missing. Check MAIL and PASS in .env")
                return False
            if not user_email:
                print("[email] Recipient email missing.")
                return False
            email_content = self._read_email_template()
            send_ok = self._send_email(user_email, email_content)
            print(f"[email] Alert send status: {send_ok}")
            return send_ok
        else:
            print("Suicide percentage below threshold. No action taken.")
            return False

if __name__=="__main__":
    # Replace with your real email credentials
    load_dotenv()
    
    sender_email = os.getenv("MAIL")
    sender_password = os.getenv("PASS")
    print(sender_email, sender_password)

    monitor = MentalHealthMonitor(sender_email, sender_password)

    label_counts = {
        'stress': 2,
        'anxiety': 1,
        'suicide': 7,
        'ptsd': 0
    }

    user_email = 'awwab.mahimi.0074@gmail.com'

    monitor.evaluate_and_notify(label_counts, user_email)
