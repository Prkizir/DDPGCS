obs = rlNumericSpec([3 1],...
    'LowerLimit',[-inf -inf 0  ]',...
    'UpperLimit',[ inf  inf inf]');
obs.Name = 'observations';
obs.Description = 'integrated error, error, and measured temperature';
obs_num = obs.Dimension(1);

act = rlNumericSpec([1 1]);
act.Name = 'tmp';
act_num = act.Dimension(1);

env = rlSimulinkEnv('RLmodel','RLmodel/RL Temperature Agent',...
    obs,act);

env.ResetFcn = @(in)localResetFcn(in);

Ts = 1.0;
Tf = 200;

rng(0)

state_representation = [
    featureInputLayer(obs_num,'Normalization','none','Name','State')
    fullyConnectedLayer(50,'Name','sfc1')
    reluLayer('Name','CriticRelu1')
    fullyConnectedLayer(25,'Name','sfc2')];
action_representation = [
    featureInputLayer(act_num,'Normalization','none','Name','Action')
    fullyConnectedLayer(25,'Name','cfc1')];
common = [
    additionLayer(2,'Name','add')
    reluLayer('Name','ccrl')
    fullyConnectedLayer(1,'Name','Output')];

cn = layerGraph();
cn = addLayers(cn,state_representation);
cn = addLayers(cn,action_representation);
cn = addLayers(cn,common);
cn = connectLayers(cn,'sfc2','add/in1');
cn = connectLayers(cn,'cfc1','add/in2');

figure
plot(cn)

critic_opt = rlRepresentationOptions('LearnRate',1e-03,...
    'GradientThreshold',1);
critic = rlQValueRepresentation(cn,obs,act,'Observation',{'State'},...
    'Action',{'Action'},critic_opt);

actor_network = [
    featureInputLayer(obs_num,'Normalization','none','Name','State')
    fullyConnectedLayer(3, 'Name','afc1')
    tanhLayer('Name','at')
    fullyConnectedLayer(act_num,'Name','Action')
    ];

actor_opt = rlRepresentationOptions('LearnRate',1e-04,...
    'GradientThreshold',1);

actor = rlDeterministicActorRepresentation(actor_network,obs,act,...
    'Observation',{'State'},'Action',{'Action'},actor_opt);

agent_opt = rlDDPGAgentOptions(...
    'SampleTime',Ts,...
    'TargetSmoothFactor',1e-3,...
    'DiscountFactor',1.0, ...
    'MiniBatchSize',64, ...
    'ExperienceBufferLength',1e6); 
agent_opt.NoiseOptions.Variance = 0.3;
agent_opt.NoiseOptions.VarianceDecayRate = 1e-5;

temperature_agent = rlDDPGAgent(actor,critic,agent_opt);
maxepisodes = 5000;
maxsteps = ceil(Tf/Ts);
train_opt = rlTrainingOptions(...
    'MaxEpisodes',maxepisodes, ...
    'MaxStepsPerEpisode',maxsteps, ...
    'ScoreAveragingWindowLength',20, ...
    'Verbose',false, ...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',800);

training_stats = train(temperature_agent, env, train_opt);

sim_opt = rlSimulationOptions('MaxSteps',maxsteps,'StopOnError','on');
experiences = sim(env,temperature_agent,sim_opt);

function in = localResetFcn(in)

% randomize reference signal
blk = sprintf('RLmodel/Reference Temperature');
h = 4*randn + 20;
while h <= 15 || h >= 30
    h = 4*randn + 20;
end
in = setBlockParameter(in,blk,'Value',num2str(h));

end