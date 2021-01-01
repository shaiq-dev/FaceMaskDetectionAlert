import enum


class EmailHost(enum.Enum):
    GMAIL = {'server': 'smtp.gmail.com', 'port': 587}
    OUTLOOK = {'server': 'smtp-mail.outlook.com', 'port': 587}
    OTHER = False


class EmailConfiguration(object):
    def __init__(self,
                 from_email,
                 to_email,
                 from_email_password,
                 host=EmailHost.GMAIL,
                 host_config=None):

        self.__from_email = from_email
        self.__to_email = to_email if isinstance(to_email,
                                                 list) else list(to_email)
        self.__from_email_password = from_email_password

        if host is EmailHost.OTHER:
            try:
                self.__server = host_config['server']
                self.__port = host_config['port']
            except Exception as _e:
                raise ValueError("Invalid Host Config")
        else:
            self.__server = host.value['server']
            self.__port = host.value['port']

    def get_from_email(self):
        return self.__from_email

    def get_to_email(self):
        return self.__to_email

    def get_password(self):
        return str(self.__from_email_password)

    def get_server(self):
        return self.__server

    def get_port(self):
        return self.__port
