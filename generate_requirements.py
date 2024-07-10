import subprocess

def freeze_requirements():
    with open('requirements.txt', 'w') as f:
        result = subprocess.run(['pip', 'freeze'], stdout=subprocess.PIPE, text=True)
        f.write(result.stdout)

if __name__ == "__main__":
    freeze_requirements()
