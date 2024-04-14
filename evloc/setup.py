from setuptools import find_packages, setup
import glob

package_name = 'evloc'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Mover los archivos de resources a el directorio install
        (f'share/{package_name}/resources/local_clouds', glob.glob('resources/local_clouds/*')),
        (f'share/{package_name}/resources', glob.glob('resources/map_global_ori.ply')),
        (f'share/{package_name}/resources', glob.glob('resources/map_global_sim.pcd')),
        (f'share/{package_name}/resources', glob.glob('resources/map_global_sim_empty.pcd')),
        (f'share/{package_name}/resources', glob.glob('resources/groundtruth_data.csv'))
  
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='GonzaloVega',
    maintainer_email='g.vega.2020@alumnos.urjc.es',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "evloc_node = evloc.evloc_node:main"
        ],
    },
)
