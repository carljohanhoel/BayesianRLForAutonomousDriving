from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Lambda, add, Input, Reshape, Conv1D, MaxPooling1D, concatenate
import keras.backend as K


class NetworkMLP(object):
    """
    This class is used to build a neural network with an MLP structure.

    There are different functions that builds a standard MLP, w/wo dueling architecture,
    and w/wo additional untrainable prior network.

    Args:
        nb_inputs (int): Number of inputs to the network.
        nb_outputs (int): Number of outputs from the network.
        nb_hidden_layers (int): Number of hidden layers.
        nb_hidden_neurons (int): Number of neurons in the hidden layers.
        duel (bool): Use dueling architecture.
        prior (bool): Use an additional untrainable prior network.
        prior_scale_factor (float): Scale factor that balances trainable/untrainable contribution to the output.
        duel_type (str): 'avg', 'max', or 'naive'
        activation (str): Type of activation function, see Keras for definition
        window_length (int): How many historic states that are used as input. Set to 1 in this work.
    """
    def __init__(self, nb_inputs, nb_outputs, nb_hidden_layers, nb_hidden_neurons, duel, prior, prior_scale_factor=10.,
                 duel_type='avg', activation='relu', window_length=1):
        self.model = None
        if not prior and not duel:
            self.build_mlp(nb_inputs, nb_outputs, nb_hidden_layers, nb_hidden_neurons, activation=activation,
                           window_length=window_length)
        elif not prior and duel:
            self.build_mlp_dueling(nb_inputs, nb_outputs, nb_hidden_layers, nb_hidden_neurons, dueling_type=duel_type,
                                   activation=activation, window_length=window_length)
        elif prior and not duel:
            self.build_prior_plus_trainable(nb_inputs, nb_outputs, nb_hidden_layers, nb_hidden_neurons,
                                            activation=activation, prior_scale_factor=prior_scale_factor,
                                            window_length=window_length)
        elif prior and duel:
            self.build_prior_plus_trainable_dueling(nb_inputs, nb_outputs, nb_hidden_layers, nb_hidden_neurons,
                                                    dueling_type=duel_type, activation=activation,
                                                    prior_scale_factor=prior_scale_factor, window_length=window_length)
        else:
            raise Exception('Error in Network creation')

    def build_mlp(self, nb_inputs, nb_outputs, nb_hidden_layers, nb_hidden_neurons, activation='relu', window_length=1):
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(window_length, nb_inputs)))
        for _ in range(nb_hidden_layers):
            self.model.add(Dense(nb_hidden_neurons))
            self.model.add(Activation(activation))
        self.model.add(Dense(nb_outputs, activation='linear'))

    def build_mlp_dueling(self, nb_inputs, nb_outputs, nb_hidden_layers, nb_hidden_neurons, dueling_type='avg',
                          activation='relu', window_length=1):
        self.build_mlp(nb_inputs, nb_outputs, nb_hidden_layers, nb_hidden_neurons, activation=activation,
                       window_length=window_length)
        layer = self.model.layers[-2]
        y = Dense(nb_outputs + 1, activation='linear')(layer.output)
        if dueling_type == 'avg':
            outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True),
                                 output_shape=(nb_outputs,))(y)
        elif dueling_type == 'max':
            outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.max(a[:, 1:], keepdims=True),
                                 output_shape=(nb_outputs,))(y)
        elif dueling_type == 'naive':
            outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:], output_shape=(nb_outputs,))(y)
        else:
            assert False, "dueling_type must be one of {'avg','max','naive'}"
        self.model = Model(inputs=self.model.input, outputs=outputlayer)

    def build_prior_plus_trainable(self, nb_inputs, nb_outputs, nb_hidden_layers, nb_hidden_neurons, activation='relu',
                                   prior_scale_factor=1., window_length=1):
        net_input = Input(shape=(window_length, nb_inputs), name='input')

        prior_net = Flatten()(net_input)
        for _ in range(nb_hidden_layers):
            prior_net = Dense(nb_hidden_neurons, activation=activation, kernel_initializer='glorot_normal',
                              trainable=False)(prior_net)
        prior_out = Dense(nb_outputs, activation='linear', trainable=False, name='prior_out')(prior_net)
        prior_scale = Lambda(lambda x: x * prior_scale_factor, name='prior_scale')(prior_out)

        trainable_net = Flatten(input_shape=(window_length, nb_inputs))(net_input)
        for _ in range(nb_hidden_layers):
            trainable_net = Dense(nb_hidden_neurons, activation=activation, kernel_initializer='glorot_normal',
                                  trainable=True)(trainable_net)
        trainable_out = Dense(nb_outputs, activation='linear', trainable=True, name='trainable_out')(trainable_net)

        add_output = add([trainable_out, prior_scale], name='add')

        self.model = Model(inputs=net_input, outputs=add_output)

    def build_prior_plus_trainable_dueling(self, nb_inputs, nb_outputs, nb_hidden_layers, nb_hidden_neurons,
                                           activation='relu', prior_scale_factor=1., dueling_type='avg',
                                           window_length=1):
        net_input = Input(shape=(window_length, nb_inputs), name='input')

        prior_net = Flatten()(net_input)
        for _ in range(nb_hidden_layers):
            prior_net = Dense(nb_hidden_neurons, activation=activation, kernel_initializer='glorot_normal',
                              trainable=False)(prior_net)
        prior_out_wo_dueling = Dense(nb_outputs + 1, activation='linear', trainable=False,
                                     name='prior_out_wo_dueling')(prior_net)
        if dueling_type == 'avg':
            prior_out = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True),
                               output_shape=(nb_outputs,), name='prior_out')(prior_out_wo_dueling)
        elif dueling_type == 'max':
            prior_out = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.max(a[:, 1:], keepdims=True),
                               output_shape=(nb_outputs,), name='prior_out')(prior_out_wo_dueling)
        elif dueling_type == 'naive':
            prior_out = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:], output_shape=(nb_outputs,),
                               name='prior_out')(prior_out_wo_dueling)
        else:
            assert False, "dueling_type must be one of {'avg','max','naive'}"
        prior_scale = Lambda(lambda x: x * prior_scale_factor, name='prior_scale')(prior_out)

        trainable_net = Flatten(input_shape=(window_length, nb_inputs))(net_input)
        for _ in range(nb_hidden_layers):
            trainable_net = Dense(nb_hidden_neurons, activation=activation, kernel_initializer='glorot_normal',
                                  trainable=True)(trainable_net)
        trainable_out_wo_dueling = Dense(nb_outputs + 1, activation='linear', trainable=True,
                                         name='trainable_out_wo_dueling')(trainable_net)
        if dueling_type == 'avg':
            trainable_out = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True),
                                   output_shape=(nb_outputs,), name='trainable_out')(trainable_out_wo_dueling)
        elif dueling_type == 'max':
            trainable_out = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.max(a[:, 1:], keepdims=True),
                                   output_shape=(nb_outputs,), name='trainable_out')(trainable_out_wo_dueling)
        elif dueling_type == 'naive':
            trainable_out = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:], output_shape=(nb_outputs,),
                                   name='trainable_out')(trainable_out_wo_dueling)
        else:
            assert False, "dueling_type must be one of {'avg','max','naive'}"

        add_output = add([trainable_out, prior_scale], name='add')

        self.model = Model(inputs=net_input, outputs=add_output)


class NetworkCNN(object):
    """
    This class is used to build a neural network with a CNN structure.

    There are different functions that builds a standard CNN, w/wo dueling architecture,
    and w/wo additional untrainable prior network.

    Args:
        nb_ego_states (int): Number of states that describe the ego vehicle.
        nb_states_per_vehicle (int): Number of states that describe each of the surrounding vehicles.
        nb_vehicles (int): Maximum number of surrounding vehicles.
        nb_actions: (int): Number of outputs from the network.
        nb_conv_layers (int): Number of convolutional layers.
        nb_conv_filters (int): Number of convolutional filters.
        nb_hidden_fc_layers (int): Number of hidden layers.
        nb_hidden_neurons (int): Number of neurons in the hidden layers.
        duel (bool): Use dueling architecture.
        prior (bool): Use an additional untrainable prior network.
        prior_scale_factor (float): Scale factor that balances trainable/untrainable contribution to the output.
        duel_type (str): 'avg', 'max', or 'naive'
        activation (str): Type of activation function, see Keras for definition
        window_length (int): How many historic states that are used as input. Set to 1 in this work.
    """
    def __init__(self, nb_ego_states, nb_states_per_vehicle, nb_vehicles, nb_actions, nb_conv_layers, nb_conv_filters,
                 nb_hidden_fc_layers, nb_hidden_neurons, duel, prior, prior_scale_factor=10., duel_type='avg',
                 activation='relu', window_length=1):
        self.model = None
        if not prior and not duel:
            self.build_cnn(nb_ego_states, nb_states_per_vehicle, nb_vehicles, nb_actions, nb_conv_layers,
                           nb_conv_filters, nb_hidden_fc_layers, nb_hidden_neurons, activation=activation,
                           window_length=window_length)
        elif not prior and duel:
            self.build_cnn_dueling(nb_ego_states, nb_states_per_vehicle, nb_vehicles, nb_actions, nb_conv_layers,
                                   nb_conv_filters, nb_hidden_fc_layers, nb_hidden_neurons, dueling_type=duel_type,
                                   activation=activation, window_length=window_length)
        elif prior and duel:
            self.build_cnn_dueling_prior(nb_ego_states, nb_states_per_vehicle, nb_vehicles, nb_actions, nb_conv_layers,
                                         nb_conv_filters, nb_hidden_fc_layers, nb_hidden_neurons,
                                         dueling_type=duel_type, activation=activation,
                                         prior_scale_factor=prior_scale_factor, window_length=window_length)
        else:
            raise Exception('Error in Network creation')

    def build_cnn(self, nb_ego_states, nb_states_per_vehicle, nb_vehicles, nb_actions, nb_conv_layers, nb_conv_filters,
                  nb_hidden_fc_layers, nb_hidden_neurons, activation='relu', window_length=1):
        nb_inputs = nb_ego_states + nb_states_per_vehicle * nb_vehicles

        net_input = Input(shape=(window_length, nb_inputs), name='input')
        flat_input = Flatten()(net_input)

        input_ego = Lambda(lambda state: state[:, :nb_ego_states * window_length])(flat_input)
        input_others = Lambda(lambda state: state[:, nb_ego_states * window_length:])(flat_input)
        input_others_reshaped = Reshape((nb_vehicles * nb_states_per_vehicle * window_length, 1,),
                                        input_shape=(nb_vehicles * nb_states_per_vehicle *
                                                     window_length,))(input_others)

        conv_net = Conv1D(nb_conv_filters, nb_states_per_vehicle*window_length,
                          strides=nb_states_per_vehicle*window_length, activation=activation,
                          kernel_initializer='glorot_normal')(input_others_reshaped)
        for _ in range(nb_conv_layers-1):
            conv_net = Conv1D(nb_conv_filters, 1, strides=1, activation=activation,
                              kernel_initializer='glorot_normal')(conv_net)
        pool = MaxPooling1D(pool_size=nb_vehicles)(conv_net)
        conv_net_out = Reshape((nb_conv_filters,), input_shape=(1, nb_conv_filters,), name='convnet_out')(pool)

        merged = concatenate([input_ego, conv_net_out])

        joint_net = Dense(nb_hidden_neurons, activation=activation, kernel_initializer='glorot_normal')(merged)
        for _ in range(nb_hidden_fc_layers-1):
            joint_net = Dense(nb_hidden_neurons, activation=activation, kernel_initializer='glorot_normal')(joint_net)

        output = Dense(nb_actions, activation='linear', name='output')(joint_net)

        self.model = Model(inputs=net_input, outputs=output)

    def build_cnn_dueling(self, nb_ego_states, nb_states_per_vehicle, nb_vehicles, nb_actions, nb_conv_layers,
                          nb_conv_filters, nb_hidden_fc_layers, nb_hidden_neurons, activation='relu', window_length=1,
                          dueling_type='avg'):
        self. build_cnn(nb_ego_states=nb_ego_states, nb_states_per_vehicle=nb_states_per_vehicle,
                        nb_vehicles=nb_vehicles, nb_actions=nb_actions, nb_conv_layers=nb_conv_layers,
                        nb_conv_filters=nb_conv_filters, nb_hidden_fc_layers=nb_hidden_fc_layers,
                        nb_hidden_neurons=nb_hidden_neurons, activation=activation, window_length=window_length)
        layer = self.model.layers[-2]
        y = Dense(nb_actions + 1, activation='linear')(layer.output)
        if dueling_type == 'avg':
            outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True),
                                 output_shape=(nb_actions,))(y)
        elif dueling_type == 'max':
            outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.max(a[:, 1:], keepdims=True),
                                 output_shape=(nb_actions,))(y)
        elif dueling_type == 'naive':
            outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:], output_shape=(nb_actions,))(y)
        else:
            assert False, "dueling_type must be one of {'avg','max','naive'}"
        self.model = Model(inputs=self.model.input, outputs=outputlayer)

    def build_cnn_dueling_prior(self, nb_ego_states, nb_states_per_vehicle, nb_vehicles, nb_actions, nb_conv_layers,
                                nb_conv_filters, nb_hidden_fc_layers, nb_hidden_neurons, activation='relu',
                                window_length=1, dueling_type='avg', prior_scale_factor=1.):
        nb_inputs = nb_ego_states + nb_states_per_vehicle * nb_vehicles

        net_input = Input(shape=(window_length, nb_inputs), name='input')
        flat_input = Flatten()(net_input)
        input_ego = Lambda(lambda state: state[:, :nb_ego_states * window_length])(flat_input)
        input_others = Lambda(lambda state: state[:, nb_ego_states * window_length:])(flat_input)
        input_others_reshaped = Reshape((nb_vehicles * nb_states_per_vehicle * window_length, 1,),
                                        input_shape=(nb_vehicles * nb_states_per_vehicle *
                                                     window_length,))(input_others)

        prior_conv_net = Conv1D(nb_conv_filters, nb_states_per_vehicle * window_length,
                                strides=nb_states_per_vehicle * window_length, activation=activation,
                                kernel_initializer='glorot_normal', trainable=False)(input_others_reshaped)
        for _ in range(nb_conv_layers - 1):
            prior_conv_net = Conv1D(nb_conv_filters, 1, strides=1, activation=activation,
                                    kernel_initializer='glorot_normal', trainable=False)(prior_conv_net)
        prior_pool = MaxPooling1D(pool_size=nb_vehicles)(prior_conv_net)
        prior_conv_net_out = Reshape((nb_conv_filters,), input_shape=(1, nb_conv_filters,),
                                     name='prior_convnet_out')(prior_pool)
        prior_merged = concatenate([input_ego, prior_conv_net_out])
        prior_joint_net = Dense(nb_hidden_neurons, activation=activation, kernel_initializer='glorot_normal',
                                trainable=False)(prior_merged)
        for _ in range(nb_hidden_fc_layers-1):
            prior_joint_net = Dense(nb_hidden_neurons, activation=activation, kernel_initializer='glorot_normal',
                                    trainable=False)(prior_joint_net)
        prior_out_wo_dueling = Dense(nb_actions+1, activation='linear', name='prior_out_wo_dueling',
                                     trainable=False)(prior_joint_net)
        if dueling_type == 'avg':
            prior_out = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True),
                               output_shape=(nb_actions,), name='prior_out')(prior_out_wo_dueling)
        elif dueling_type == 'max':
            prior_out = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.max(a[:, 1:], keepdims=True),
                               output_shape=(nb_actions,), name='prior_out')(prior_out_wo_dueling)
        elif dueling_type == 'naive':
            prior_out = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:],
                               output_shape=(nb_actions,), name='prior_out')(prior_out_wo_dueling)
        else:
            assert False, "dueling_type must be one of {'avg','max','naive'}"
        prior_scale = Lambda(lambda x: x * prior_scale_factor, name='prior_scale')(prior_out)

        trainable_conv_net = Conv1D(nb_conv_filters, nb_states_per_vehicle * window_length,
                                    strides=nb_states_per_vehicle * window_length, activation=activation,
                                    kernel_initializer='glorot_normal', trainable=True)(input_others_reshaped)
        for _ in range(nb_conv_layers - 1):
            trainable_conv_net = Conv1D(nb_conv_filters, 1, strides=1, activation=activation,
                                        kernel_initializer='glorot_normal', trainable=True)(trainable_conv_net)
        trainable_pool = MaxPooling1D(pool_size=nb_vehicles)(trainable_conv_net)
        trainable_conv_net_out = Reshape((nb_conv_filters,), input_shape=(1, nb_conv_filters,),
                                         name='trainable_convnet_out')(trainable_pool)
        trainable_merged = concatenate([input_ego, trainable_conv_net_out])
        trainable_joint_net = Dense(nb_hidden_neurons, activation=activation, kernel_initializer='glorot_normal',
                                    trainable=True)(trainable_merged)
        for _ in range(nb_hidden_fc_layers-1):
            trainable_joint_net = Dense(nb_hidden_neurons, activation=activation, kernel_initializer='glorot_normal',
                                        trainable=True)(trainable_joint_net)
        trainable_out_wo_dueling = Dense(nb_actions + 1, activation='linear', name='trainable_out_wo_dueling',
                                         trainable=True)(trainable_joint_net)
        if dueling_type == 'avg':
            trainable_out = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True),
                                   output_shape=(nb_actions,), name='trainable_out')(trainable_out_wo_dueling)
        elif dueling_type == 'max':
            trainable_out = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.max(a[:, 1:], keepdims=True),
                                   output_shape=(nb_actions,), name='trainable_out')(trainable_out_wo_dueling)
        elif dueling_type == 'naive':
            trainable_out = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:],
                                   output_shape=(nb_actions,), name='trainable_out')(trainable_out_wo_dueling)
        else:
            assert False, "dueling_type must be one of {'avg','max','naive'}"

        add_output = add([trainable_out, prior_scale], name='final_output')

        self.model = Model(inputs=net_input, outputs=add_output)
