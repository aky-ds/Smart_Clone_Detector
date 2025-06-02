from setuptools import setup,find_packages
def get_requires(file_path:str):
    requires=[]
    with open(file_path) as f:
        requires=f.readlines()
        requires=[require.replace('\n','') for require in requires]
        if '-e .' in requires:
            requires.remove('-e .')
    return requires

setup(
    
    name='Smart Clone Detector',
    version=0.1,
    author='Ayazulhaq Yousafzai',
    author_email='www.ayazkhan.com.21@gmail.com',
    description='End to End Smart DeepFake Detector on Heruku',
    install_requires=get_requires('Requirements.txt'),
    packages=find_packages()
    
)