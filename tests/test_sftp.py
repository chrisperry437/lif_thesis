import paramiko

HOST = "192.168.1.103"
PORT = 22
USERNAME = "Rapid-E-user"
PASSWORD = "QEYKvnnw"

transport = paramiko.Transport((HOST, PORT))
transport.connect(
    username=USERNAME,
    password=PASSWORD
)

sftp = paramiko.SFTPClient.from_transport(transport)

print("Connected!")

print("\nData folder contents:")

for item in sftp.listdir("/DATA/D_00001"):
    print(item)

sftp.close()
transport.close()