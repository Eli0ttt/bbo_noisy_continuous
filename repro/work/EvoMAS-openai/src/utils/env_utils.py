"""
Environment and repository utilities for local SWE-bench evaluation.
"""

import contextlib
import glob
import os
import shutil
import subprocess
from os.path import dirname as pdirname
from os.path import join as pjoin
from pathlib import Path


@contextlib.contextmanager
def cd(newdir):
    """
    Context manager for changing the current working directory
    :param newdir: path to the new directory
    :return: None
    """
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


def run_command(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """
    Run a command in the shell.
    Args:
        - cmd: command to run
    """
    try:
        cp = subprocess.run(cmd, check=True, **kwargs)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}, {e}")
        raise e
    return cp


def repo_commit_current_changes():
    """
    Commit the current active changes so that it's safer to do git reset later on.
    Use case: for storing the changes made in pre_install and test_patch in a commit.
    """
    add_all_cmd = ["git", "add", "."]
    commit_cmd = [
        "git",
        "commit",
        "--allow-empty",
        "-m",
        "Temporary commit for storing changes",
    ]
    run_command(add_all_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    run_command(commit_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def repo_clean_changes() -> None:
    """
    Reset repo to HEAD. Basically clean active changes and untracked files on top of HEAD.

    A lightweight version of `repo_reset_and_clean_checkout`. This is also used in different scenarios.
    """
    reset_cmd = ["git", "reset", "--hard"]
    clean_cmd = ["git", "clean", "-fd"]
    run_command(reset_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    run_command(clean_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def repo_reset_and_clean_checkout(commit_hash: str) -> None:
    """
    Run commands to reset repo to the original commit state.
    Cleans both the uncommited changes and the untracked files, and submodule changes.
    Assumption: The current directory is the git repository.
    """
    # Clean files that might be in .gitignore, but could have been created by previous runs
    if os.path.exists(".coverage"):
        os.remove(".coverage")
    if os.path.exists("tests/.coveragerc"):
        os.remove("tests/.coveragerc")
    other_cov_files = glob.glob(".coverage.TSS.*", recursive=True)
    for f in other_cov_files:
        os.remove(f)

    reset_cmd = ["git", "reset", "--hard", commit_hash]
    clean_cmd = ["git", "clean", "-fd"]
    checkout_cmd = ["git", "checkout", commit_hash]
    run_command(reset_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    run_command(clean_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # need to checkout before submodule init. Otherwise submodule may init to another version
    run_command(checkout_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # this is a fail-safe combo to reset any changes to the submodule: first unbind all submodules
    # and then make a fresh checkout of them.
    # Reference: https://stackoverflow.com/questions/10906554/how-do-i-revert-my-changes-to-a-git-submodule
    submodule_unbind_cmd = ["git", "submodule", "deinit", "-f", "."]
    submodule_init_cmd = ["git", "submodule", "update", "--init"]
    run_command(
        submodule_unbind_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    run_command(
        submodule_init_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def run_script_in_conda(
    args: list[str], env_name: str, **kwargs
) -> subprocess.CompletedProcess:
    """
    Run a python command in a given conda environment.
    """
    cmd = ["conda", "run", "-n", env_name, "python", *args]
    return subprocess.run(cmd, **kwargs)


def run_string_cmd_in_conda(
    command: str, env_name: str, **kwargs
) -> subprocess.CompletedProcess:
    """
    Run a complete command in a given conda environment, where the command is a string.

    This is useful when the command to be run contains &&, etc.

    NOTE: use `conda activate` instead of `conda run` in this verison, so that we can
          run commands that contain `&&`, etc.
    """
    conda_bin_path = os.getenv("CONDA_EXE")  # for calling conda
    if conda_bin_path is None:
        raise RuntimeError("Env variable CONDA_EXE is not set")
    conda_root_dir = pdirname(pdirname(conda_bin_path))
    conda_script_path = pjoin(conda_root_dir, "etc", "profile.d", "conda.sh")
    command = command.replace("'", '"')
    conda_cmd = f"bash -c 'source {conda_script_path} ; conda activate {env_name} ; {command} ; conda deactivate'"
    print(f"Running command: {conda_cmd}")
    return subprocess.run(conda_cmd, shell=True, **kwargs)


def create_dir_if_not_exists(dir_path: str):
    """
    Create a directory if it does not exist.
    Args:
        dir_path (str): Path to the directory.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def create_fresh_dir(dir_path: str):
    """
    Create a dir from fresh.
    If there is existing dir with the same name, remove it.
    Args:
        dir_path (str): Path to the directory.
    """
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)
