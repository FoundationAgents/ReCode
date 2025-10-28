from gym.envs.registration import register

from envs.webshop.src.webshop.web_agent_site.envs.web_agent_site_env import WebAgentSiteEnv
from envs.webshop.src.webshop.web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

register(
  id='WebAgentSiteEnv-v0',
  entry_point='envs.webshop.src.webshop.web_agent_site.envs:WebAgentSiteEnv',
)

register(
  id='WebAgentTextEnv-v0',
  entry_point='envs.webshop.src.webshop.web_agent_site.envs:WebAgentTextEnv',
)