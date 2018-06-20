require 'cutorch'
require 'cunn'
require 'nngraph'

return function(args)
  local in_row_s = 65
  local in_col = 300
  local in_hist = 4
  local input_dims_s = {in_hist,in_row_s,in_col}
  local region_hight = {1,2,3} -- hight only of filter, width will be 'in_col'

  local n_filters = args.AEN_n_filters --or 20-- number of filters per region size
  local tot_filters_s = table.getn(region_hight)*n_filters

  local output_size = args.n_objects
  local net_s = nn.Sequential()
  net_s:add(nn.Reshape(unpack(input_dims_s)))
  local net_concat = nn.Concat(2)
  for i=1,table.getn(region_hight) do
    local net = nn.Sequential()
    net:add(nn.SpatialConvolution(4,n_filters,in_col,region_hight[i]))
    net:add(nn.ReLU())
    net:add(nn.SpatialDropout(0.5))
    net:add(nn.SpatialMaxPooling(1,in_row_s-region_hight[i]+1))
    net_concat:add(net)
  end
  net_s:add(net_concat)
  net_s:add(nn.Reshape(tot_filters_s))
  net_s:add(nn.Linear(tot_filters_s,output_size))
  if args.shallow_elimination_flag ==0 then
    net_s:add(nn.Sigmoid())
  end

  print("AEN:\n", net_s)
  return net_s
end
