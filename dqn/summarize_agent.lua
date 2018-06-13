require 'plot'

cmd = torch.CmdLine()
cmd:option("-agent",'',"name of agent to summarize")
cmd:option("-title",'',"agent title")
cmd:option("-limit",1000,"[0..] maximum number of points")
cmd:option("-AEN_pref",false,"plot AEN figure")
cmd:option("-refresh",false,"refresh summary")
params = cmd:parse(arg)
if params.title == '' then  params.title = nil end
plotAgent(params.agent,params.limit,params.title,params.AEN_pref,params.refresh)
