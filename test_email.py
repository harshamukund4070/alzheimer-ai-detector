import os
import smtplib
from dotenv import load_dotenv

load_dotenv()
EMAIL = os.environ.get('EMAIL_HOST_USER')
PASS = os.environ.get('EMAIL_HOST_PASSWORD')

try:
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.set_debuglevel(1)
    server.starttls()
    server.login(EMAIL, PASS)
    print("LOGIN SUCCESS")
    server.quit()
except Exception as e:
    import traceback
    traceback.print_exc()
