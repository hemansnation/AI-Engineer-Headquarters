from pathlib import Path

DOCS = {
    'setup.md': '''# Orbit X200 setup guide
## First time setup
Plug in the power adapter and connect the blue cable from the modem to the WAN port. Wait two minutes until the status light turns solid white. Then open the Orbit app and scan the QR code on the bottom of the router.
## Reset your admin password
If you forgot the admin password, open the Orbit app, tap Sign in, then tap Forgot password. We send a six digit code to the email on the account. Enter the code and pick a new password.
## Factory reset
To bring the router back to factory settings, press and hold the recessed reset button for 10 seconds with a paperclip. The light blinks amber, then white. All Wi-Fi names, passwords and port rules are erased.
## Update the firmware
Open the app, go to Settings, then Firmware. Tap Check for update. The router restarts twice and takes about six minutes. Do not unplug it during the update.
''',
    'errors.md': '''# Orbit X200 error codes
## Error E-4012
E-4012 means the WAN link is down. The light on the front turns solid red and no website loads. Check the blue cable between the modem and the router, then restart the modem for 30 seconds.
## Error E-5103
E-5103 means DNS lookup failed. Websites do not open but the router still shows a green light. Switch the DNS setting to automatic in Settings, then Network.
## Error E-7781
E-7781 means the firmware file failed its checksum. The download was damaged. Delete the update in the app and download it again on a stable connection.
## Replacement parts
The power adapter is part PWR-9128-B. The older adapter PWR-9128-A does not fit the X200. The external antenna is part ANT-2210.
''',
    'warranty.md': '''# Orbit warranty and returns
## Warranty
Every Orbit router has a 24 month limited warranty that starts on the day you buy it. The warranty covers hardware faults. It does not cover water damage or a cracked case.
## How to claim
Open a claim in the Orbit app under Help, then Warranty. Upload a photo of the serial number on the bottom of the router. We reply in two working days and send a prepaid return label.
## Returns
You can return the router within 30 days if you do not like it. The box does not need to be sealed. Refunds reach your card in five to seven working days.
''',
}


def main():
    target = Path('data') / 'docs'
    target.mkdir(parents=True, exist_ok=True)
    for name, text in DOCS.items():
        (target / name).write_text(text, encoding='utf-8')
        print(f'wrote {target / name}')


if __name__ == '__main__':
    main()
