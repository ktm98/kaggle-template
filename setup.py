from setuptools import setup, find_packages


def _requires_from_file(filename):
    return open(filename).read().splitlines()


# see
# https://qiita.com/jkawamoto/items/32a57be3cf7b10c18d50
def take_package_name(name):
    if name.startswith("-e"):
        return name[name.find("=")+1:name.rfind("-")]
    else:
        return name.strip()

def load_requires_from_file(filepath):
    with open(filepath) as fp:
        return [take_package_name(pkg_name) for pkg_name in fp.readlines()]

def load_links_from_file(filepath):
    res = []
    with open(filepath) as fp:
        for pkg_name in fp.readlines():
            if pkg_name.startswith("-e"):
                res.append(pkg_name.split(" ")[1])
    return res


setup(
    name='kaggle_templete',
    version='0.0.1',
    packages=find_packages(),
    # install_requires=_requires_from_file('requirements.txt'),
    description='utility scripts for kaggle',
    dependency_links=load_links_from_file("requirements.txt"),
    install_requires=load_requires_from_file("requirements.txt"),

    author='ktm',
)