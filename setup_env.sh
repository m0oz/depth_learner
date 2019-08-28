venv_dir=./venv2.7

# Create virtualenv if not existent
if [ ! -d "$venv_dir" ]; then
    python -m pip install --user virtualenv
    python -m virtualenv --system-site-packages --python="/usr/bin/python2.7" "$venv_dir"
    source "$venv_dir/bin/activate"
    pip install -r requirements.txt
fi

source "$venv_dir/bin/activate"

# Source ros bash files (Update with your ros path and catkin ws!!)
# Possibly add this to your .bashrc
source "/opt/ros/kinetic/setup.bash"
source "/home/rpg_students/moritz/ros_catkin_ws/devel/setup.bash"
