%=========================%
%    Environment Model    %
%=========================%

% Observation Data Structure

obs = rlNumericSpec([3 1],...
    'LowerLimit',[-inf -inf 0]',...
    'UpperLimit',[inf inf inf]');

obs.Name = 'observations';
obs.Description = 'integrated error, error, and measured temperature';
obs_num = obs.Dimension(1);

% Action Data Structure
act = rlNumericSpec([1 1]);
act.Name = 'tmp';
act_num = act.Dimension(1);

% Environment Interface

env = rlSimulinkEnv('DDPGEnv','DDPGEnv/DDPG Temperature Agent',...
    obs,act);

% Environment Reset Function
env.ResetFcn = @(in)localResetFcn(in);

% Timestep and Total Run Time
Ts = 1.0;
Tf = 200;

% Random Seed = 0
rng(0)

%========================%
%    DDPG Agent Model    %
%========================%

%========================%
%         Critic         %
%========================%

% Critic State Path
state_representation = [
    featureInputLayer(obs_num, 'Normalization', 'none', 'Name', 'State')
    fullyConnectedLayer(75,'Name','sfc1')
    reluLayer('Name','sr1')
    fullyConnectedLayer(50,'Name','sfc2')
    reluLayer('Name','sr2')
    fullyConnectedLayer(25,'Name','sfc3')];

% Critic Action Path
action_representation = [
    featureInputLayer(act_num, 'Normalization', 'none', 'Name', 'Action')
    fullyConnectedLayer(25,'Name','cfc1')];

% Critic Common Path
common = [
    additionLayer(2, 'Name', 'add')
    reluLayer('Name', 'ccr1')
    fullyConnectedLayer(1, 'Name', 'Output')];

% Critic Network
cn = layerGraph();
cn = addLayers(cn, state_representation);
cn = addLayers(cn, action_representation);
cn = addLayers(cn, common);
cn = connectLayers(cn, 'sfc3','add/in1');
cn = connectLayers(cn, 'cfc1','add/in2');

% Plot Network
figure
plot(cn)

% Critic Options
critic_opt = rlRepresentationOptions('LearnRate', 1e-03,...
    'GradientThreshold', 1, 'UseDevice', "gpu");

% Critic
critic = rlQValueRepresentation(cn, obs, act, 'Observation', {'State'},...
    'Action', {'Action'}, critic_opt);

%=======================%
%         Actor         %
%=======================%

% Actor Network
actor_network = [
    featureInputLayer(obs_num, 'Normalization', 'none', 'Name', 'State')
    fullyConnectedLayer(3, 'Name', 'afc1')
    tanhLayer('Name','at')
    fullyConnectedLayer(act_num, 'Name', 'Action')];

% Actor Options
actor_opt = rlRepresentationOptions('LearnRate', 1e-04,...
    'GradientThreshold', 1, 'UseDevice', "gpu");

% Actor
actor = rlDeterministicActorRepresentation(actor_network, obs, act,...
    'Observation', {'State'}, 'Action', {'Action'}, actor_opt);

%=======================%
%      Create Agent     %
%=======================%

% Agent Options
agent_opt = rlDDPGAgentOptions(...
    'SampleTime', Ts, ...
    'TargetSmoothFactor', 1e-3, ...
    'DiscountFactor', 1.0, ...
    'MiniBatchSize', 64, ...
    'ExperienceBufferLength', 1e6);

agent_opt.NoiseOptions.Variance = 0.3;
agent_opt.NoiseOptions.VarianceDecayRate = 1e-5;

% Agent 1
temperature_agent = rlDDPGAgent(actor, critic, agent_opt);

%======================%
%      Train Agent     %
%======================%

maxepisodes = 10000;
maxsteps = ceil(Tf/Ts);

train_opt = rlTrainingOptions(...
    'MaxEpisodes', maxepisodes, ...
    'MaxStepsPerEpisode', maxsteps, ...
    'ScoreAveragingWindowLength', 20, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'StopTrainingCriteria', 'AverageReward', ...
    'StopTrainingValue', 800);

training_stats = train(temperature_agent, env, train_opt);
sim_opt = rlSimulationOptions('MaxSteps', maxsteps, 'StopOnError', 'on');
experiences = sim(env, temperature_agent, sim_opt);


function in = localResetFcn(in)
    
    % Reference Temperature
    block = sprintf('DDPGEnv/Reference Temperature');
    h = 4*randn + 20;
    
    while h <= 15 || h >= 30
        h = 4*randn + 20;
    end
    
    in = setBlockParameter(in, block, 'Value', num2str(h));
end

