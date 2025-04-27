# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from stretch.agent.base import ManagedOperation


class GoToStart(ManagedOperation):
    """Put the robot into navigation mode"""

    location = None
    _successful = False

    def __init__(self, *args, location, **kwargs):
        super().__init__(*args, **kwargs)
        self.location = location 
        self.plan = None

    def configure(self, location):
        self.location = location

    def can_start(self) -> bool:
        print(
            f"{self.name}: check to see if path to start is possible."
        )
        self.plan = None

        end = self.location
        start = self.robot.get_base_pose()
        
        if not self.navigation_space.is_valid(start):
            self.error(
                "Robot is in an invalid configuration. It is probably too close to geometry, or localization has failed."
            )

        plan = self.agent.planner.plan(start, end)

        if plan.success:
            self.plan = plan
            self.cheer("Can go to Start!")
            return True
        self.error("START Planning failed!")
        return False

    def run(self) -> None:
        self.intro(f"Attempting move to {self.location}")
        self.agent.robot_say("Failed my task, moving to start base location.")
        # self.agent.robot_say("Moving to start base location.")
        self.robot.move_to_nav_posture()
        # Execute the trajectory
        assert (
            self.plan is not None
        ), "Did you make sure that we had a plan? You should call can_start() before run()."
        self.robot.execute_trajectory(self.plan, final_timeout=10.0)
        self._successful = True
        self.cheer(f"Done attempting moving to {self.location}")

    def was_successful(self) -> bool:
        return self._successful
