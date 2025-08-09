import os
import sys
import subprocess
import json


def create_vscode_files(venv_dir):
    """
    Create VS Code configuration files including settings.json, launch.json, and a workspace file.

    Args:
        venv_dir (str): Path to the virtual environment directory.

    Creates:
        - .vscode/settings.json: Contains Python and editor settings.
        - .vscode/launch.json: Configuration for launching Python files.
        - project.code-workspace: VS Code workspace configuration.

    Notes:
        - Uses correct python interpreter path on Windows/Linux/macOS.
    """
    print("\n[8] Creating VS Code files: settings, launch, workspace")

    vscode_dir = os.path.join(os.getcwd(), ".vscode")
    settings_path = os.path.join(vscode_dir, "settings.json")
    launch_path = os.path.join(vscode_dir, "launch.json")
    workspace_path = os.path.join(os.getcwd(), "project.code-workspace")

    os.makedirs(vscode_dir, exist_ok=True)

    # Cross-platform interpreter path inside the venv
    interp = (
        os.path.join(venv_dir, "Scripts", "python.exe")
        if os.name == "nt"
        else os.path.join(venv_dir, "bin", "python")
    )

    # Create settings.json
    settings = {
        "python.defaultInterpreterPath": interp,
        "python.terminal.activateEnvironment": True,
        "editor.formatOnSave": True,
        "python.formatting.provider": "black",
    }
    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)

    # Create launch.json
    launch = {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "Run main.py",
                "type": "python",
                "request": "launch",
                "program": "${workspaceFolder}/main.py",
                "console": "integratedTerminal",
                "justMyCode": True,
            }
        ],
    }
    with open(launch_path, "w", encoding="utf-8") as f:
        json.dump(launch, f, indent=2)

    # Create project.code-workspace
    workspace = {
        "folders": [{"path": "."}],
        "settings": {
            "python.defaultInterpreterPath": interp,
        },
    }
    with open(workspace_path, "w", encoding="utf-8") as f:
        json.dump(workspace, f, indent=2)

    print(
        "VS Code files created successfully: settings.json, launch.json, project.code-workspace"
    )


def load_or_create_config():
    """
    Load or create the setup-config.json file.

    Returns:
        dict: The configuration loaded from or written to setup-config.json.
    """
    print("\n[1] Setting up setup-config.json")
    config_path = os.path.join(os.getcwd(), "setup-config.json")

    if not os.path.exists(config_path):
        print("Creating config file...")
        default_config = {
            "project_name": os.path.basename(os.getcwd()),
            "main_file": "app.py",
            "entry_point": "main.py",
            "requirements_file": "requirements.txt",
            "venv_dir": "venv",
            "python_version": "3.12",
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=2)
        print("setup-config.json created.")
    else:
        print("Config file already exists.")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_virtualenv(venv_dir, python_version=None):
    """
    Create a virtual environment using the current Python interpreter.

    Args:
        venv_dir (str): Directory path for the virtual environment.
        python_version (str, optional): Target Python version (unused).
    """
    print("\n[2] Checking virtual environment")
    if not os.path.exists(venv_dir):
        print(f"Creating virtual environment at: {venv_dir}")
        subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
        print("Virtual environment created.")
    else:
        print("Virtual environment already exists.")


def create_requirements_file(path):
    """
    Create a default requirements.txt if it doesn't exist.

    Args:
        path (str): File path for requirements.txt.
    """
    print("\n[3] Checking requirements.txt")
    if not os.path.exists(path):
        print("Creating empty requirements.txt...")
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Add your dependencies here\n")
        print("requirements.txt created.")
    else:
        print("requirements.txt already exists.")


def install_requirements(venv_dir, requirements_path):
    """
    Install packages from requirements.txt.

    Args:
        venv_dir (str): Directory path for the virtual environment.
        requirements_path (str): Path to requirements.txt file.
    """
    print("\n[4] Installing requirements")
    pip_path = (
        os.path.join(venv_dir, "Scripts", "pip.exe")
        if os.name == "nt"
        else os.path.join(venv_dir, "bin", "pip")
    )
    subprocess.run([pip_path, "install", "-r", requirements_path], check=True)
    print("Packages installed.")


def upgrade_pip(venv_dir):
    """
    Upgrade pip inside the virtual environment.

    Args:
        venv_dir (str): Directory path for the virtual environment.
    """
    print("\n[5] Upgrading pip")
    python_path = (
        os.path.join(venv_dir, "Scripts", "python.exe")
        if os.name == "nt"
        else os.path.join(venv_dir, "bin", "python")
    )
    subprocess.run([python_path, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    print("pip upgraded.")


def create_env_info(venv_dir):
    """
    Save basic information about the virtual environment.

    Args:
        venv_dir (str): Directory path for the virtual environment.
    """
    print("\n[6] Creating env-info.txt")
    info_path = "env-info.txt"
    python_path = (
        os.path.join(venv_dir, "Scripts", "python.exe")
        if os.name == "nt"
        else os.path.join(venv_dir, "bin", "python")
    )
    with open(info_path, "w", encoding="utf-8") as f:
        subprocess.run([python_path, "--version"], stdout=f)
        f.write("\nInstalled packages:\n")
        subprocess.run([python_path, "-m", "pip", "freeze"], stdout=f)
    print(f"Environment info saved to {info_path}")


def create_main_file(main_file_path, venv_dir):
    """
    Create main.py with virtualenv activation and dynamic script execution.

    Args:
        main_file_path (str): Path to main.py file.
        venv_dir (str): Directory path for the virtual environment.
    """
    print("\n[7] Checking main.py")
    if not os.path.exists(main_file_path):
        print(f"Creating {main_file_path}...")

        # Build the embedded script with dynamic site-packages detection
        main_code = f'''
import os
import sys
import json

# Activate the virtual environment (embedded)
base_dir = os.path.dirname(__file__)
if os.name == "nt":
    venv_site_packages = os.path.join(base_dir, r"{venv_dir}", "Lib", "site-packages")
else:
    pyver = "python" + str(sys.version_info[0]) + "." + str(sys.version_info[1])
    venv_site_packages = os.path.join(base_dir, r"{venv_dir}", "lib", pyver, "site-packages")

if os.path.isdir(venv_site_packages) and venv_site_packages not in sys.path:
    sys.path.insert(0, venv_site_packages)

print("Virtual environment activated within script.")
print("sys.path includes:", venv_site_packages)

# Load configuration from config file for flexibility
CONFIG_FILE = "setup-config.json"

def load_entry_point():
    if not os.path.exists(CONFIG_FILE):
        print(f"{{CONFIG_FILE}} not found.")
        sys.exit(1)

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = json.load(f)

    main_file = config.get("main_file", "app.py")
    if not os.path.exists(main_file):
        print(f"{{main_file}} does not exist.")
        sys.exit(1)

    print(f"Running: {{main_file}}")
    with open(main_file, encoding="utf-8") as f:
        code = compile(f.read(), main_file, "exec")
        exec(code, globals())

if __name__ == "__main__":
    load_entry_point()
'''.lstrip()

        with open(main_file_path, "w", encoding="utf-8") as f:
            f.write(main_code)
        print(f"{main_file_path} created.")
    else:
        print("main.py already exists.")


def create_app_file(app_file_path):
    """
    Create a simple app.py file with a welcome message if it does not exist.
    Args:
        app_file_path (str): Path to app.py or the main application file.
    """
    print("\n[7.1] Checking app.py")
    if not os.path.exists(app_file_path):
        print(f"Creating {app_file_path} with welcome message...")
        welcome_code = '''\
print("Welcome! This is your app.py file.")
print("You can now start writing your application code here.")
'''
        with open(app_file_path, "w", encoding="utf-8") as f:
            f.write(welcome_code)
        print(f"{app_file_path} created.")
    else:
        print(f"{app_file_path} already exists.")


if __name__ == "__main__":
    print("\nStarting project setup...\n" + "-" * 40)
    config = load_or_create_config()

    venv_dir = config["venv_dir"]
    requirements_path = config["requirements_file"]
    main_file = config["main_file"]
    python_version = config["python_version"]

    create_virtualenv(venv_dir, python_version)
    create_requirements_file(requirements_path)
    # Prefer upgrading pip before installing deps
    upgrade_pip(venv_dir)
    install_requirements(venv_dir, requirements_path)
    create_env_info(venv_dir)

    entry_point = config.get("entry_point", "main.py")
    create_main_file(entry_point, venv_dir)

    create_app_file(main_file)
    create_vscode_files(venv_dir)

    print("\nProject setup complete.")
