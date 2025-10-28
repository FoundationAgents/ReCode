import warnings
warnings.filterwarnings("ignore", "The 'text' argument to find\\(\\)-type methods is deprecated", category=DeprecationWarning)

from typing import Any, Optional
from bs4 import BeautifulSoup
from bs4.element import Comment

from utils.common import read_json_file

from base.environment import Env
from envs.webshop.src.webshop.web_agent_site.envs import WebAgentTextEnv
from envs.webshop.src.webshop.web_agent_site.utils import DEFAULT_FILE_PATH
from envs.webshop.src.webshop.web_agent_site.envs.web_agent_text_env import SimServer

_SHARED_SERVER = None

def clean_str(p):
    """Clean string encoding issues"""
    return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")

def tag_visible(element):
    """Check if HTML element should be visible in text conversion"""
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return (
        element.parent.name not in ignore and not isinstance(element, Comment)
    )

def webshop_text(html_content, max_products=10):
    """Convert WebShop HTML to text with proper formatting.
    
    Args:
        html_content: HTML content to parse
        max_products: Maximum number of products to display (default: 10)
    """
    try:
        # Parse HTML content
        html_obj = BeautifulSoup(html_content, 'html.parser')
        texts = html_obj.find_all(string=True)
        visible_texts = list(filter(tag_visible, texts))
        
        # Format text
        observation = ''
        option_type = ''
        option_types = {}
        asins = []
        cnt = 0
        prod_cnt = 0
        just_prod = 0
        
        for t in visible_texts:
            if t == '\n': 
                continue
            if t.replace('\n', '').replace('\\n', '').replace(' ', '') == '': 
                continue
                
            if t.parent.name == 'button':  # button
                processed_t = f'\n[{t}] '
            elif t.parent.name == 'label':  # options
                processed_t = f'[{t}]'
                option_types[str(t)] = option_type
            elif t.parent.get('class') == ["product-link"]:  # product asins
                processed_t = f'\n[{t}] '
                if prod_cnt >= max_products:
                    processed_t = ''
                prod_cnt += 1
                asins.append(str(t))
                just_prod = 0
            else:  # regular, unclickable text
                processed_t = '\n' + str(t) + ' '
                if cnt < 2: 
                    processed_t = ''
                if just_prod <= 2 and prod_cnt >= max_products + 1: 
                    processed_t = ''
                option_type = str(t)
                cnt += 1
            just_prod += 1
            observation += processed_t
        
        # Build info dict
        info = {}
        if option_types:
            info['option_types'] = option_types
        if asins:
            info['asins'] = asins
        if 'Your score (min 0.0, max 1.0)' in visible_texts:
            idx = visible_texts.index('Your score (min 0.0, max 1.0)')
            info['reward'] = float(visible_texts[idx + 1])
            observation = 'Your score (min 0.0, max 1.0): ' + str(visible_texts[idx + 1])
        
        return clean_str(observation), info
        
    except Exception as e:
        # Fallback to basic format if parsing fails
        return f"HTML parsing error: {str(e)}", {}

# def _read_first_non_ws_char(file_path: Path) -> Optional[str]:
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             chunk = f.read(2048)
#             for ch in chunk:
#                 if not ch.isspace():
#                     return ch
#     except Exception:
#         return None
#     return None

def _get_shared_server(
    file_path: Optional[str],
    num_products: Optional[int],
    human_goals: bool,
    limit_goals: int = -1,
    quiet: bool = False,
):
    global _SHARED_SERVER
    if _SHARED_SERVER is None:
        _SHARED_SERVER = SimServer(
            base_url='http://127.0.0.1:3000',
            file_path=file_path,
            filter_goals=None,
            limit_goals=limit_goals,
            num_products=num_products,
            human_goals=human_goals,
            show_attrs=False,
            quiet=quiet,
        )
    return _SHARED_SERVER

class WebShopEnv(Env):
    """WebShop Environment for agent interaction."""
    env_name = "webshop"
    def __init__(
        self,
        logger: Optional[Any] = None,
        max_steps: int = 30,
        file_path: Optional[str] = DEFAULT_FILE_PATH,
        success_threshold: float = 1.0,
    ):
        self.logger = logger
        self.max_steps = max_steps
        self.success_threshold = success_threshold
        
        # Initialize environment state
        self.id = "webshop_env"
        self._step_count = 0
        self.is_finished = False
        self.reward = 0.0
        self.last_observation = ""
        self.last_raw_observation = ""  # Store raw observation
        self.current_session = None
        self.trajectory = []  # Store complete trajectory like human agent
        


        # Use a shared SimServer to avoid reloading data per instance
        self._server = _get_shared_server(
            file_path=file_path,
            num_products=None,
            human_goals=True,
            quiet=True,
        )

        self.webshop_env = WebAgentTextEnv(
            observation_mode="text",
            server=self._server,
            num_products=None,
            human_goals=True,
            quiet=True,
        )
        
        if self.logger:
            self.logger.info(f"WebShop environment initialized")
            self.logger.info(f"Configuration: max_steps={self.max_steps}, "
                           f"success_threshold={self.success_threshold}")
    
    
    def _ensure_session_asins(self):
        """Ensure user session has proper asins field (fix for product click bug)"""
        session_id = self.webshop_env.session
        if hasattr(self.webshop_env, 'server') and session_id in self.webshop_env.server.user_sessions:
            session = self.webshop_env.server.user_sessions[session_id]
            if 'asins' not in session:
                session['asins'] = set()
            elif not isinstance(session['asins'], set):
                # Convert list to set if needed
                session['asins'] = set(session['asins']) if hasattr(session['asins'], '__iter__') else set()
    
    def reset(self, running_config: dict, id: Optional[str] = None):
        """Reset the environment using official WebShop reset"""
        if self.logger:
            self.logger.info(f"Resetting WebShop environment (ID: {id})")
            
        self._step_count = 0
        self.is_finished = False
        self.reward = 0.0
        self.trajectory = []  # Reset trajectory

        self.split = running_config.get("split", "train")

        if self.split == "train":
            self.indices = read_json_file(f"envs/webshop/data/train_indices.json")
        elif self.split == "test":
            self.indices = read_json_file(f"envs/webshop/data/test_indices.json")
        else:
            raise ValueError(f"Invalid split: {self.split}. WebShop has only train and test splits.")
        
        # Use official WebShop reset
        self.id = id
        id_int: Optional[int] = None

        if id is not None:
            try:
                id_int = int(id)
            except ValueError:
                raise ValueError(f"Task ID '{id}' is not a valid integer.")

        self.session_id = self.indices[id_int]
        result = self.webshop_env.reset(session=self.session_id)
        
        # Handle both tuple (observation, info) and single observation return
        if isinstance(result, tuple):
            observation, info = result
        else:
            observation = result
            info = {}
        
        # Ensure user session has proper asins field
        self._ensure_session_asins()
        
        # Store raw observation first
        self.last_raw_observation = observation if observation else "No observation available"
        
        # Now format for agent
        formatted_observation = self._format_observation(observation)
        self.last_observation = formatted_observation
        self.current_session = self.webshop_env.session
        
        # Create trajectory entry in human agent style
        trajectory_entry = {
            "action": None,  # No action for reset
            "observation": formatted_observation,
            "raw_observation": self.last_raw_observation,
            "url": "http://127.0.0.1:3000",  # WebShop base URL
            "goal": self.webshop_env.get_instruction_text(),
            "step": self._step_count,
            "session": self.current_session,
            "reward": 0.0,
            "info": info
        }
        self.trajectory.append(trajectory_entry)
        
        if self.logger:
            self.logger.info(f"WebShop reset with session {self.current_session}")
        
        return {"observations": [formatted_observation], "env_name": self.env_name, "env": self}
    
    def _format_observation(self, observation: str) -> str:
        """Format observation with proper text conversion."""
        if observation is None:
            return "No observation available"
        
        try:
            if hasattr(self.webshop_env, 'state') and self.webshop_env.state.get('html'):
                html_content = self.webshop_env.state['html']
                formatted_obs, info = webshop_text(html_content)
                
                if info and hasattr(self.webshop_env.server, 'user_sessions') and self.current_session:
                    current_session_info = self.webshop_env.server.user_sessions.get(self.current_session, {})
                    current_session_info.update(info)
                
                if formatted_obs and formatted_obs.strip():
                    return formatted_obs
        except Exception:
            pass
        
        return observation.replace(' [SEP] ', '\n')
    
    async def _run(self, action: str):
        """Execute an action using official WebShop step function"""
        if self.is_finished:
            if self.logger:
                self.logger.warning(f"Attempted action '{action}' on finished environment")
            return self.last_observation
        
        self._step_count += 1
        
        if self.logger:
            self.logger.info(f"Step {self._step_count}: {action}")
        
        # Handle [FINISH] action - this terminates the episode
        if action == "[FINISH]":
            self.is_finished = True
            if self.logger:
                self.logger.info(f"[FINISH] action received - terminating episode")
                self.logger.info(f"Episode finished with final reward: {self.reward:.3f}, success: {self.is_success()}")
            
            # Add finish action to trajectory
            trajectory_entry = {
                "action": action,
                "observation": self.last_observation,
                "raw_observation": self.last_raw_observation,
                "url": "FINISH",
                "goal": self.webshop_env.get_instruction_text(),
                "step": self._step_count,
                "session": self.current_session,
                "reward": self.reward,
                "info": {"finish_action": True}
            }
            self.trajectory.append(trajectory_entry)
            
            return self.last_observation
        
        # Check for step limit before execution
        if self._step_count > self.max_steps:
            self.is_finished = True
            if self.logger:
                self.logger.info(f"Maximum steps ({self.max_steps}) reached. Episode terminated.")
            return self.last_observation
        
        # Use official WebShop step function
        try:
            # Ensure user session has proper asins field
            self._ensure_session_asins()
            
            result = self.webshop_env.step(action)
            
            # Handle different return formats
            if isinstance(result, tuple) and len(result) >= 4:
                observation, reward, done, info = result[:4]
            elif hasattr(result, '__iter__') and len(list(result)) >= 4:
                observation, reward, done, info = list(result)[:4]
            else:
                # Fallback for unexpected return format
                observation = str(result) if result is not None else "No observation available"
                reward, done, info = 0.0, False, {}
            
            # Ensure observation is not None
            if observation is None:
                observation = "No observation available"
            
            # Store raw observation first
            self.last_raw_observation = observation
            
            # Format observation for agent
            formatted_observation = self._format_observation(observation)
            self.last_observation = formatted_observation
            self.reward = reward if reward is not None else 0.0
            
            # Add to trajectory in human agent style
            try:
                goal = self.webshop_env.get_instruction_text()
            except:
                goal = "Episode completed"
            
            trajectory_entry = {
                "action": action,
                "observation": formatted_observation,
                "raw_observation": self.last_raw_observation,
                "url": self._get_current_url(),
                "goal": goal,
                "step": self._step_count,
                "session": self.current_session,
                "reward": self.reward,
                "info": info if info is not None else {}
            }
            self.trajectory.append(trajectory_entry)
            
            # WebShop has its own done condition when user clicks "Buy Now"
            self.is_finished = done or self._step_count >= self.max_steps
            
            return formatted_observation
            
        except Exception as e:
            self.is_finished = True
            error_msg = f"WebShop step execution failed: {str(e)}"
            
            # Add error to trajectory
            try:
                goal = self.webshop_env.get_instruction_text()
            except:
                goal = "Error occurred"
            
            trajectory_entry = {
                "action": action,
                "observation": error_msg,
                "raw_observation": error_msg,
                "url": "ERROR",
                "goal": goal,
                "step": self._step_count,
                "session": self.current_session,
                "reward": 0.0,
                "info": {"error": str(e)}
            }
            self.trajectory.append(trajectory_entry)
            
            if self.logger:
                self.logger.error(f"Step {self._step_count} ERROR: {error_msg}")
            
            return error_msg
    
    def _get_current_url(self):
        """Get current URL based on WebShop state - simplified version"""
        # This is a simplified version since we don't have direct access to WebShop's internal URL state
        # In the real WebShop, this would be more detailed
        base_url = "http://127.0.0.1:3000"
        
        # Try to infer URL from observation content
        if "search" in self.last_observation.lower():
            return f"{base_url}/search"
        elif "product" in self.last_observation.lower() or "Buy Now" in self.last_observation:
            return f"{base_url}/item"
        else:
            return base_url
    
    def is_done(self):
        """Check if the episode is done"""
        return self.is_finished or self._step_count >= self.max_steps
    
    def is_success(self):
        """Check if the task was completed successfully"""
        success = self.is_finished and self.reward >= self.success_threshold
        
        if self.logger and self.is_finished:
            self.logger.info(f"Task evaluation: reward={self.reward:.3f}, "
                           f"threshold={self.success_threshold}, success={success}")
        
        return success
    
    def get_step_count(self):
        """Get the current step count"""
        return self._step_count
    
    def get_reward(self):
        """Get the current reward"""
        return self.reward
    
    def get_available_actions(self):
        """Get available actions from official WebShop environment"""
        return self.webshop_env.get_available_actions()
    
    def get_instruction_text(self):
        """Get current instruction text from official WebShop environment"""
        return self.webshop_env.get_instruction_text()
    
    def get_trajectory(self):
        """Get the complete trajectory in human agent format"""
        return self.trajectory
    
    async def close(self) -> None:
        """Close the official WebShop environment"""
        if self.logger:
            self.logger.info(f"Closing WebShop environment. "
                           f"Final stats: {len(self.trajectory)} trajectory steps, "
                           f"final reward: {self.reward:.3f}, success: {self.is_success()}")
            
            # Log trajectory summary for debugging
            if self.trajectory:
                self.logger.info(f"Trajectory summary:")
                for i, step in enumerate(self.trajectory):
                    action = step.get('action', 'RESET')
                    reward = step.get('reward', 0.0)
                    url = step.get('url', 'unknown')
                    self.logger.info(f"  {i}: '{action}' -> {url} -> reward={reward:.3f}")
        
        try:
            # Close the WebShop environment
            if hasattr(self, 'webshop_env') and self.webshop_env:
                self.webshop_env.close()
            
            # Clean up shared server if it's the last instance
            global _SHARED_SERVER
            if _SHARED_SERVER is not None:
                try:
                    _SHARED_SERVER = None
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Error cleaning up shared server: {e}")
            
            # Reset state variables
            self._step_count = 0
            self.is_finished = False
            self.reward = 0.0
            self.trajectory = []
            self.current_session = None
            self.last_observation = ""
            self.last_raw_observation = ""
            
            if self.logger:
                self.logger.info("WebShop environment closed successfully")
                
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error closing WebShop environment: {e}")
            raise

    def report(self):
        return {
            "success": self.is_success(),
            "reward": self.reward,
            "step": self._step_count,
        }