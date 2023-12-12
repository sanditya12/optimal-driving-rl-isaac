import gym
from gym import spaces
import numpy as np
import math
import carb
import copy
from racing_data.center_points import center_points
from racing_data.left_boundary_points import left_boundary_points
from racing_data.right_boundary_points import right_boundary_points
import utils
from time import time
import csv

from reference_generator import ReferenceGenerator
class RacingEnv(gym.Env):
    def __init__(
            self,
            skip_frame=1,
            physics_dt=1.0 / 60.0,
            rendering_dt=1.0 / 60.0,
            max_episode_length=2048,
            seed=0,
            headless=True,
            identifier = "",
            training_timesteps = 0
    ) -> None:
        from omni.isaac.kit import SimulationApp
        self.headless = headless
        self.simulation_app = SimulationApp({"headless" : self.headless, "anti_aliasing": 0})
        self.skip_frame = skip_frame
        self.dt = physics_dt * self.skip_frame
        self.max_episode_length = max_episode_length
        self.steps_after_reset = int(rendering_dt/physics_dt)
        self.identifier = identifier
        self.training_timesteps = training_timesteps
        
        from omni.isaac.core import World
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        from omni.isaac.wheeled_robots.robots import WheeledRobot
        from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
        from omni.isaac.debug_draw import _debug_draw
        from omni.isaac.core.utils.rotations import quat_to_euler_angles
        self.draw = _debug_draw.acquire_debug_draw_interface()
        self.get_euler_angle = quat_to_euler_angles
        

        self.world = World(physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return  
        


        #Init Jetbot
        jetbot_asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
        self.pos_init_xyz = np.array([-1.64, -1.62, 0])
        self.jetbot = self.world.scene.add(
            WheeledRobot(
                prim_path="/World/Jetbot",
                name="jetbot",
                wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                create_robot=True,
                usd_path=jetbot_asset_path,
                position = self.pos_init_xyz
            )
        )
        self.jetbot_controller = DifferentialController(name="simple_control", wheel_radius=0.0325, wheel_base=0.1125)

        # set seed
        self.seed(seed)

        #RL Constraints
        self.reward_range = (-float("inf"), float("inf"))
        self.action_space = spaces.Box(low=-1, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-float("inf"), high=float("inf"), shape=(20,), dtype=np.float32)

        #Robot Constraints
        self.max_velocity = 1
        self.min_velocity = 0.5
        self.max_angular_velocity = math.pi
        self.max_ray_length = 3
        self.rob_radius = 0.07
        self.max_distance_from_center = 0.365
                
        
        self.vision_horzion = 10
        center_points_2d = [sublist[:2] for sublist in center_points]
        self.rg = ReferenceGenerator(horizon = self.vision_horzion, center_points= center_points_2d)
        self.count = 0
        self.n_collisions = 0

        self.cat_states = []
        self.cat_forwards = []
        self.training_cat_states = []
        self.reset_count = 0

        self.lap_time = 0
        self.lap_timestep = 0

        self.n_collisions_100k= 0

        self.n_collisions_200k= 0

        # left_boundary_segments = utils.convert_points_to_segments(left_boundary_points)
        # right_boundary_segments= utils.convert_points_to_segments(right_boundary_points)

        self.finish_line = utils.get_boundary_for_point(center_points_2d[-2], center_points_2d[-1],None, self.max_distance_from_center*2)
        # print(self.finish_line)
        return
    
    def get_dt(self):
        return self._dt
    
    def step(self,action):
        self.draw_track()

        # APPLYING ACTION
        raw_forward = action[0]
        raw_angular = action[1]
        forward_velocity = ((raw_forward + 1) / 2) * (self.max_velocity - self.min_velocity) + self.min_velocity
        angular_velocity = raw_angular * self.max_angular_velocity
        u_desired = np.array([forward_velocity, angular_velocity])

        state_init = self.get_jetbot_state()
        visible_center_points = self.rg.generate_map((state_init[0], state_init[1]))
        x_ref, y_ref = visible_center_points[-1]
        visible_center_points_1d = np.array(visible_center_points).flatten()

        u = u_desired

        self.draw.draw_points([(x_ref, y_ref, 0)],[(1,1,0,1)], [7])

        for i in range(self.skip_frame):
            self.jetbot.apply_wheel_actions(
                self.jetbot_controller.forward(command=u)
            )
            self.world.step(render=False)
        # END APPLYING ACTION

        velocities = self.jetbot.get_linear_velocity()
        # print("velocity: ",velocities[0])


        # current_jetbot_position, _ = self.jetbot.get_world_pose()
        state_current = self.get_jetbot_state()
        rays, self.obs = self.get_rays(state_current[0],state_current[1],state_current[2])

        observations= self.get_observations()
        info={}
        done = False

        self.draw.clear_lines()

        map_points = []
        for ray in rays:
            # print(ray)
            ray_xyz = ([sublist+[0] for sublist in ray])
            self.draw.draw_lines([ray_xyz[0]], [ray_xyz[1]], [(0, 0, 0, 1)],[1])
            map_points.append(ray[1])



        prev_position = np.array([state_init[0], state_init[1]])
        curr_position = np.array([state_current[0], state_current[1]])
        ref_position = np.array([x_ref, y_ref])
        #CHECK MAX EPISODE LENGTH
        if self.world.current_time_step_index - self.steps_after_reset>= self. max_episode_length:
            # print("Done because of time")
            done = True

        previous_dist_to_ref = np.linalg.norm(ref_position - prev_position)
        current_dist_to_ref = np.linalg.norm(ref_position - curr_position)
    
        reward = (previous_dist_to_ref - current_dist_to_ref)*10
        # print(reward)

        # CHECK FINISH LINE
        if utils.is_circle_intersecting_with_line(curr_position, self.rob_radius, self.finish_line[0], self.finish_line[1]):
            print("Time for one lap: ", self.world.current_time_step_index - self.steps_after_reset)
            reward = 1
            done = True
            self.lap_time = time()-self.start_time
            self.lap_timestep = self.world.current_time_step_index - self.steps_after_reset
                
        self.draw.clear_points()
        color = (0,1,0,1)

        if utils.is_circle_intersecting_with_points(curr_position, self.rob_radius, map_points):
            self.n_collisions += 1
            # print("Collides! number of collisions so far: ", self.n_collisions)
            reward = -1 
            done = True

        if(self.count %10 == 0):
            self.cat_states.append(curr_position)
            self.cat_forwards.append(forward_velocity)
        self.count += 1
        # print(self.world.current_time_step_index)

        if(self.count == 100000):
            self.n_collisions_100k = self.n_collisions
        
        if(self.count == 200000):
            self.n_collisions_200k = self.n_collisions

        return observations, reward, done, info
    
    def reset(self):
        self.world.reset()
        self.reset_counter = 0
        self.start_time = time()
        if(self.reset_count % 10 == 0):
            self.training_cat_states.append(self.cat_states)
        
        self.cat_states=[]
        self.cat_forwards=[]
        x,y,th = self.get_jetbot_state()
        _, self.obs = self.get_rays(x, y ,th)

        visible_center_points = self.rg.generate_map((x, y))
        x_ref, y_ref = visible_center_points[-1]
        x_bef_ref, y_bef_ref = visible_center_points[-2]
        self.ref_line = self.rg.get_boundary_for_point((x_bef_ref, y_bef_ref),(x_ref,y_ref), None, 0.37*2)
        self.reset_count+= 1
        observations = self.get_observations()

        return observations

    def get_observations(self):
        self.world.render()

        return np.concatenate(
            [
                self.obs
            ]
        )
    
    def render(self, mode="human"):
        return
    
    def close(self):
        with open("testing_data/"+self.identifier+".csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "Number of Collisions",
                    self.n_collisions
                ]
            )
            writer.writerow([
                "Lap Time",
                self.lap_time,
            ])
            writer.writerow([
                "Lap Timestep",
                self.lap_timestep
            ])
            writer.writerow([
                "Cat Positions",
                self.get_formatted_cat_states(self.cat_states)
            ])
            writer.writerow([
                "Cat Speed",
                self.get_formatted_cat_forwards()
            ])

        if self.training_timesteps!=0:
            training_trajectories = "["
            for i, st in enumerate(self.training_cat_states):
                training_trajectories+= self.get_formatted_cat_states(st)+ ","
            training_trajectories+= "]"
            with open("training_data/"+self.identifier+".csv", "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    [
                        "Total Timestep (Training Duration)",
                        self.training_timesteps
                    ])
                writer.writerow(  
                    [
                        "Number of Collisions",
                        self.n_collisions
                    ]
                )
                writer.writerow(  
                    [
                        "Number of Collisions 100k",
                        self.n_collisions_100k
                    ]
                )
                writer.writerow(  
                    [
                        "Number of Collisions 200k",
                        self.n_collisions_200k
                    ]
                )
                writer.writerow(
                    [
                        "Training Trajectories",
                        training_trajectories           
                    ]
                )
        self.simulation_app.close()
        print("number of collisions for training: ", self.n_collisions)
        return
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]

    def get_jetbot_state(self)-> np.ndarray:
        jetbot_world_position, jetbot_world_orientation = self.jetbot.get_world_pose()
        jetbot_world_theta= self.get_euler_angle(jetbot_world_orientation)[-1]
        return np.array([jetbot_world_position[0], jetbot_world_position[1], jetbot_world_theta])

    def draw_track(self)-> None:
        
        self.draw.draw_lines_spline(center_points, (1, 1, 0, 1),1, False)
        self.draw.draw_lines_spline(left_boundary_points, (0, 0, 0, 8),0, False)
        self.draw.draw_lines_spline(right_boundary_points, (0, 0, 0, 8),0, False)

        left_finish_3d, right_finish_3d = [sublist+[0] for sublist in self.finish_line]
        self.draw.draw_lines([left_finish_3d], [right_finish_3d], [(0, 1, 0, 1)],[1])

    def get_rays(self, x:float, y:float ,th:float) -> np.ndarray:
        right_boundary_points_xy = [sublist[:2] for sublist in right_boundary_points]
        left_boundary_points_xy = [sublist[:2] for sublist in left_boundary_points]
        rays = []
        obs = []
        for i in range(-10,10):
            ray = utils.create_line((x,y), th+i*math.pi/15, self.max_ray_length)
            intersection = ray[1]           
            intersection_left = utils.is_intersecting_with_points(ray, left_boundary_points_xy)
            intersection_right = utils.is_intersecting_with_points(ray, right_boundary_points_xy)
            if intersection_left and intersection_right:
                if utils.get_distance(ray[0], intersection_left)< utils.get_distance(ray[0], intersection_right):
                    intersection = intersection_left
                else:
                    intersection = intersection_right
            elif intersection_left:
                intersection = intersection_left
            elif intersection_right:
                intersection = intersection_right

            ray = [ray[0], intersection]
            observation = (self.max_ray_length - utils.get_distance(ray[0], ray[1]))/self.max_ray_length
            rays.append(ray)
            obs.append(observation)
        return rays, obs
    
    def get_formatted_cat_states(self, cat_states):
        formatted = "["
        for state in cat_states:
            formatted+="["+ str(state[0])+ ", "+ str(state[1])+ "],"
        formatted+= "]"
        return formatted
    
    def get_formatted_cat_forwards(self):
        formatted = "["
        for state in self.cat_forwards:
            formatted+=str(state)+ ","
        formatted+= "]"
        return formatted