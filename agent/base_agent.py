class BaseAgent:
    def get_action(self, obs):
        raise NotImplementedError("get_action() not implemented")

    def update(self, obs, action, reward, next_obs, done):
        raise NotImplementedError("update() not implemented")

    def save(self, path):
        raise NotImplementedError("save() not implemented")

    def load(self, path):
        raise NotImplementedError("load() not implemented")
