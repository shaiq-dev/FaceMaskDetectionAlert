import os
import datetime
import threading

from smtplib import SMTP
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

from jinja2 import Environment, FileSystemLoader

from Alerts.emailConfig import EmailConfiguration
from Alerts.emailConfig import EmailHost


class EmailAlert(object):
    def __init__(self, emailConfig):

        if not isinstance(emailConfig, EmailConfiguration):
            raise ValueError("Invalid Email Configuration")

        self.emailConfig = emailConfig

        self.__env = Environment(
            loader=FileSystemLoader('%s/templates/' %
                                    os.path.dirname(__file__)))

        self.msg = MIMEMultipart()

    def _add_attachment(self, attachment):

        try:
            with open(attachment, "rb") as eattach:

                part = MIMEBase("application", "octet-stream")
                part.set_payload(eattach.read())

                # Encode file in ASCII characters to send by email
                encoders.encode_base64(part)
                # Add header as key/value pair to attachment part
                part.add_header(
                    "Content-Disposition",
                    "attachment; filename= {}".format(attachment),
                )

                self.msg.attach(part)
        except Exception as _e:
            print("[WARN] Invalid Email Attachment, discarding attachment")

    def _send_email(self, email_body, attachment):

        self.msg['Subject'] = "FaceMask Voilation Alert"
        self.msg['From'] = self.emailConfig.get_from_email()
        self.msg['To'] = ','.join(self.emailConfig.get_to_email())

        self.msg.attach(MIMEText(email_body, 'html'))
        self._add_attachment(attachment)
        try:
            server = SMTP(self.emailConfig.get_server(),
                          self.emailConfig.get_port())
            # Encrypts the email
            context = ssl.create_default_context()
            server.starttls(context=context)
            server.login(self.emailConfig.get_from_email(),
                         self.emailConfig.get_password())
            server.sendmail(self.emailConfig.get_from_email(),
                            self.emailConfig.get_to_email(),
                            self.msg.as_string())
            print('[INFO] Alert sent')
        except Exception as e:
            print("[ERROR] Something went wrong #ET10")

    def send(self, data, attachment=None, template=None):

        if template:
            _template = self.__env.get_template(template)
        else:
            _template = self.__env.get_template('email.html')
        data.update({'year': datetime.datetime.now().year})
        out_template = _template.render(data=data)

        # Threading to send emails faster
        threading.Thread(name='email_worker',
                         target=self._send_email,
                         args=[out_template, attachment]).start()
