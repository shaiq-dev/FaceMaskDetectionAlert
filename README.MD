![FaceMask Detetction Alert](./public/Fmd.svg)

#

This FaceMask Detection CNN Model is 97% accurate in detecting face masks, in videos or images.

This app has a builtin email alert system which
can therefore be used in real-time applications which require face-mask detection for safety purposes due to the outbreak of Covid-19.

#

### Project is incomplete

#

## 🚀&nbsp; Installation and Setup

1. Setup project and virtual environment

```
$ mkdir Fmd && cd Fmd
$ mkvirtualenv .env
```

2. Activate the virtual env and clone the repo

```
$ git clone https://github.com/shaiq-dev/FaceMaskDetectionAlert.git
```

3. Change your directory to the cloned repo and install the requirements

```
$ cd FaceMaskDetectionAlert
$ pip3 install -r requirements.txt
$ cd ../
```

4. create a python file to get started. This way it will use the default model "FaceMaskDetetionAlert/Models/fmd.h5".

```py
    #Fmd/app.py
    from FaceMakDetetcionAlert import FaceMaskDetector

    fmd = FaceMaskDetector()
    fmd.start()

```

To use a custopm model

```py
    fmd = FaceMaskDetector(model='PATH TO YOUR MODEL')
    fmd.start()
```

5. How to setup the Alert System

```py
    #Fmd/app.py
    from FaceMakDetetcionAlert import FaceMaskDetector
    from FaceMakDetetcionAlert.Alerts import (
        EmailConfiguration,
        EmailHost
    )

    config = EmailConfiguration(
        'sender@gmail.com',
        'receiver@gmail.com' # or a list of emails
        'SENDER PASSWORD',
    )

    fmd = FaceMaskDetector(alerts_config=config)
    fmd.start()

```

The default config uses gmail as Host, if you are using any other email provider, you have to mention that. See the builtin hosts here

```py
    config = EmailConfiguration(
        'sender@outlook.com',
        'receiver@outlook.com'
        'SENDER PASSWORD',
        host=EmailHost.OUTLOOK
    )
```

If the host is not listed in the builtin hosts, add a host config like below

```py
    other_host = {
        'server': 'SMT SERVER ADDREESS',
        'port': PORT
    }
    config = EmailConfiguration(
        'sender@outlook.com',
        'receiver@outlook.com'
        'SENDER PASSWORD',
        host=EmailHost.OTHER
        host_config=other_host
    )

```
