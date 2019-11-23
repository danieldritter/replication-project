from SL.critic.critic import Critic
from tensorflow.keras.optimizers import Adam
from data import process

def train():
    """
    1. Process data.
    2. Train actor supervised in SL model
    3. Train critic supervised
    4. Train RL agent as a function of actor and critic weights.
    """
    pass



def train_critic():
    # TODO: rename to train_SL()
    # retrieving data
    state_inputs, prev_order_inputs, season_names = process.get_data("data/standard_no_press.jsonl")

    # initializing supervised learning model and optimizer
    model = Critic()
    optimizer = Adam(0.001)
    # Looping through each game
    for i in range(len(state_inputs)):
        # Parsing just the season(not year)
        # Not sure about these conversions
        powers_seasons = []
        # extracting seasons and powers for film
        for j in range(len(season_names[i])):
            # print(season_names[i][j][0])
            powers_seasons.append(SEASON[season_names[i][j][0]] + UNIT_POWER["AUSTRIA"])
        # print(powers_seasons)

        # casting to floats
        powers_seasons = tf.convert_to_tensor(powers_seasons,dtype=tf.float32)
        state_inputs = tf.convert_to_tensor(state_inputs[i],dtype=tf.float32)
        order_inputs = tf.convert_to_tensor(prev_order_inputs[i], dtype=tf.float32)

        with tf.GradientTape() as tape:
            # applying SL model
            values = model.call(state_inputs)
            print(value)
            loss = model.loss(values, labels)
        # optimizing
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

train_critic()
