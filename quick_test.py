import gym

env = gym.make("gym_subgoal_automata:OfficeWorldDeliverCoffee-v0",
               params={"generation": "custom", "environment_seed": 0, "hide_state_variables": True})


key_to_act = {
    "w": 0,
    "s": 1,
    "a": 2,
    "d": 3,
}

env.reset()
env.render()
while True:
    x = input()

    action = key_to_act[x]
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    print(info.get("observations", {}))
    if terminated:
        print(reward)
