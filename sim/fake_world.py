# fake_world.py
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext, SimulationCfg
import omni.usd
from typing import Callable, Optional

# ------------------------------------------------------------------
# Proxy to convert old Scene methods used by Pegasus to
# current IsaacSim InteractiveScene
# Can probably be moved to independent file
# ------------------------------------------------------------------

class SceneProxy:
    def __init__(self, scene: InteractiveScene):
        self._scene = scene
        self._objects = {}

    def add(self, obj):
        if hasattr(obj, 'prim_path'):
            self._objects[obj.prim_path] = obj
        return obj

    def remove_object(self, name: str):
        if name in self._objects:
            del self._objects[name]

    def get_object(self, name: str):
        """Get tracked object by name."""
        return self._objects.get(name)

    def reset(self):
        """Reset scene."""
        return self._scene.reset()

    def __getattr__(self, name):
        return getattr(self._scene, name)


class FakeWorld:
    def __init__(
        self,
        sim_cfg: SimulationCfg,
        scene_cfg: InteractiveSceneCfg | None = None,
        physics_dt: Optional[float] = None,
        rendering_dt: Optional[float] = None,
    ) -> None:
        if physics_dt is not None:
            sim_cfg.dt = physics_dt
        if rendering_dt is not None:
            sim_cfg.render_interval = int(rendering_dt / sim_cfg.dt)
            
        self.sim = SimulationContext(sim_cfg)

        if scene_cfg is None:
            scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)

        scene = InteractiveScene(scene_cfg)
        self._scene = SceneProxy(scene)
        
        self._is_initialized = False
        self._current_time = 0.0
        
        self._physics_callbacks = {}
        self._timeline_callbacks = {}
        self._render_callbacks = {}

    # ------------------------------------------------------------------
    # Pegasus / World compatibility API
    # Can probably be moved to independent files
    # ------------------------------------------------------------------

    @property
    def stage(self):
        return omni.usd.get_context().get_stage()

    @property
    def device(self):
        return self.sim.device

    @property
    def dt(self):
        return self.sim.get_physics_dt()
    
    @property
    def current_time(self):
        return self._current_time

    def add_physics_callback(self, name: str, callback_fn: Callable):
        self._physics_callbacks[name] = callback_fn
        return self.sim.add_physics_callback(name, callback_fn)

    def remove_physics_callback(self, name: str):
        if name in self._physics_callbacks:
            del self._physics_callbacks[name]
        return self.sim.remove_physics_callback(name)

    def add_timeline_callback(self, name: str, callback_fn: Callable):
        self._timeline_callbacks[name] = callback_fn
        return self.sim.add_timeline_callback(name, callback_fn)

    def remove_timeline_callback(self, name: str):
        if name in self._timeline_callbacks:
            del self._timeline_callbacks[name]
        return self.sim.remove_timeline_callback(name)

    def add_render_callback(self, name: str, callback_fn: Callable):
        self._render_callbacks[name] = callback_fn
        return self.sim.add_render_callback(name, callback_fn)

    def remove_render_callback(self, name: str):
        if name in self._render_callbacks:
            del self._render_callbacks[name]
        return self.sim.remove_render_callback(name)

    def is_playing(self) -> bool:
        return self.sim.is_playing()

    def is_stopped(self) -> bool:
        return self.sim.is_stopped()

    def play(self):
        if not self._is_initialized:
            self.reset()
        self.sim.play()

    def pause(self):
        self.sim.pause()

    def stop(self):
        self.sim.stop()
        self._current_time = 0.0

    def render(self):
        self.sim.render()
        for callback_fn in self._render_callbacks.values():
            callback_fn(self.dt)

    # ------------------------------------------------------------------

    @property
    def scene(self):
        return self._scene

    @property
    def simulation_context(self):
        return self.sim

    def reset(self, soft: bool = False):
        if not soft:
            self.sim.reset()
            self._current_time = 0.0
        
        self.scene.reset()
        self._is_initialized = True

    def step(self, render: bool = False):
        self._current_time += self.dt
        
        self.sim.step(render=render)

        for callback_fn in self._physics_callbacks.values():
            callback_fn(self.dt)
        
        if render:
            self.render()

    def initialize_physics(self):
        if not self._is_initialized:
            self.reset()

    def clear_all_callbacks(self):
        for name in list(self._physics_callbacks.keys()):
            self.remove_physics_callback(name)
        for name in list(self._timeline_callbacks.keys()):
            self.remove_timeline_callback(name)
        for name in list(self._render_callbacks.keys()):
            self.remove_render_callback(name)

    def clear_instance(self):
        self.clear_all_callbacks()
        self.sim.clear_instance()