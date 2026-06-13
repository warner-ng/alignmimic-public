from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm

from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
import whole_body_tracking.tasks.tracking.mdp as mdp


BIKE_URDF = "/home/warner/_projects/ResMimic/assets/bicycle_top_tube/bikered.urdf"
BIKE_MESH = "/home/warner/_projects/ResMimic/assets/bikered.stl"
BIKE_OBJECT_MOTION = (
    "/home/warner/_projects/ResMimic/assets/motions/"
    "Date03_Sub01_bike_May_31_19_34_object_upright_bikez_aligned.npz"
)


@configclass
class G1FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "pelvis"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]


@configclass
class G1FlatWoStateEstimationEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class G1FlatLowFreqEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE


@configclass
class G1FlatBikeHOIEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=sim_utils.UrdfFileCfg(
                asset_path=BIKE_URDF,
                fix_base=False,
                merge_fixed_joints=True,
                # Keep IsaacLab's default convex hull path; convex_decomposition can break PhysX scene creation.
                # collider_type="convex_decomposition",
                joint_drive=None,
                scale=(0.6, 0.6, 0.6),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=0.02,
                    rest_offset=0.0,
                ),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    retain_accelerations=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=8.0),
            ),
        )
        self.commands.motion.object_asset_name = "object"
        self.commands.motion.motion_quat_order = "xyzw"
        self.commands.motion.object_motion_file = BIKE_OBJECT_MOTION
        self.commands.motion.object_root_z_bias = 0.2
        self.commands.motion.object_root_pos_offset = (-0.65, -0.25, 0.0)
        self.commands.motion.object_root_rot_offset_deg = (-90.0, 60.0, 0.0)
        self.commands.motion.object_mesh_file = BIKE_MESH
        self.commands.motion.object_scale = 0.6
        self.commands.motion.object_point_count = 1024
        self.commands.motion.human_root_rot_offset_deg = (0.0, 0.0, 0.0)
        self.commands.motion.motion_global_rot_offset_deg = (-85.0, 90.0, 0.0)
        self.commands.motion.motion_global_pos_offset = (-3.0, 0.0, 1.0)
        self.observations.policy.object_root_state = ObsTerm(
            func=mdp.object_root_state_w, params={"command_name": "motion"}
        )
        self.observations.critic.object_root_state = ObsTerm(
            func=mdp.object_root_state_w, params={"command_name": "motion"}
        )
        # self.rewards.undesired_contacts = None
        # point cloud reward, added from resmimic by warner
        self.rewards.motion_object_point_cloud = RewTerm(
            func=mdp.motion_object_point_cloud_error_exp,
            weight=2.0,
            params={"command_name": "motion", "scale": 10.0},
        )
        # self.rewards.motion_object_point_cloud = RewTerm(
        #     func=mdp.motion_object_point_cloud_error_exp,
        #     weight=0.0,
        #     params={"command_name": "motion", "scale": 10.0},
        # )
        # self.terminations.object_far = DoneTerm(
        #     func=mdp.bad_object_point_cloud, params={"command_name": "motion", "threshold": 1.0}
        # )
        # self.terminations.object_far = None
