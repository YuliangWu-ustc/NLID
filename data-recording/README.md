## Usage

1. Place the equipment, including the beakers, the platform, robotic arm, and camera system, according to the figure in the supplemental material.
2. Connect the UR5e robot to the computer and configure the IP address of the robot.
3. Connect the RGB camera and the event camera to the computer. Set the frame rate of the RGB camera to 30 FPS.
4. Open `pouring.script` on the UR5e robot, and ensure the UR5e robot is in "Remote Control" mode.
5. An example command for recording data:

```bash
./run-record.sh $PWD/test/data 1 3
```
