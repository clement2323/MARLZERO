# Backward compatibility — existing code that imports from morris_rl.env.board
# or morris_rl.env.rules continues to work without modification.
from morris_rl.env.morris.board import *  # noqa: F401, F403
from morris_rl.env.morris.rules import *  # noqa: F401, F403
