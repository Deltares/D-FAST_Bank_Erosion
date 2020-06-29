function sim2ugrid(filename)
%SIM2UGRID convert simulation results to netCDF UGRID file
%   The function copies only the data relevant for D-FAST Bank Erosion.

simOrg = qpfopen(filename);
switch simOrg.FileType
    case 'SIMONA SDS FILE'
        [xd,yd] = waquaio(simOrg,'','dgrid');
        zb      = waquaio(simOrg,'','height');
        %
        Info    = waqua('read',simOrg,'','SOLUTION_FLOW_SEP',[]);
        nTimes  = length(Info.SimTime);
        [zw,t]  = waquaio(simOrg,'','wlvl',nTimes);
        [h,t]   = waquaio(simOrg,'','wdepth',nTimes);
        [ucx,ucy,t] = waquaio(simOrg,'','xyveloc',nTimes);
        %
        node_active = ~isnan(xd(:));
        xnode   = xd(node_active);
        ynode   = yd(node_active);
        zb      = zb(node_active);
        nnodes  = length(xnode);
        %
        nodes   = xd;
        nodes(node_active) = 1:nnodes;
        faces   = cat(3, nodes([1 1:end-1],[1 1:end-1]), ...
                         nodes(:,[1 1:end-1]), ...
                         nodes, ...
                         nodes([1 1:end-1],:));
        face_active = all(~isnan(faces),3) & ~isnan(zw);
        zw      = zw(face_active);
        h       = h(face_active);
        ucx     = ucx(face_active);
        ucy     = ucy(face_active);
        %
        faces   = reshape(faces,numel(xd),4);
        faces   = faces(face_active,:);
        nfaces  = size(faces,1);
end
%
ncid = netcdf.create([filename '_map.nc'],'NETCDF4');
Err = [];
try
    inodes = netcdf.defDim(ncid,'mesh2d_nnodes',nnodes);
    ifaces = netcdf.defDim(ncid,'mesh2d_nfaces',nfaces);
    imaxfn = netcdf.defDim(ncid,'mesh2d_nmax_face_nodes',4);
    %
    mesh = netcdf.defVar(ncid,'mesh2d','NC_DOUBLE',[]);
    netcdf.putAtt(ncid,mesh,'cf_role','mesh_topology')
    netcdf.putAtt(ncid,mesh,'topology_dimension',int32(2))
    netcdf.putAtt(ncid,mesh,'node_coordinates','mesh2d_node_x mesh2d_node_y')
    netcdf.putAtt(ncid,mesh,'face_node_connectivity','mesh2d_face_nodes')
    %
    ix = netcdf.defVar(ncid,'mesh2d_node_x','NC_DOUBLE',inodes);
    netcdf.putAtt(ncid,ix,'standard_name','projection_x_coordinate')
    netcdf.putAtt(ncid,ix,'units','m')
    netcdf.putVar(ncid,ix,xnode)
    %
    iy = netcdf.defVar(ncid,'mesh2d_node_y','NC_DOUBLE',inodes);
    netcdf.putAtt(ncid,iy,'standard_name','projection_y_coordinate')
    netcdf.putAtt(ncid,iy,'units','m')
    netcdf.putVar(ncid,iy,ynode)
    %
    ifnc = netcdf.defVar(ncid,'mesh2d_face_nodes','NC_INT',[imaxfn, ifaces]);
    netcdf.putAtt(ncid,ifnc,'cf_role','face_node_connectivity')
    netcdf.putAtt(ncid,ifnc,'start_index',int32(1))
    netcdf.putVar(ncid,ifnc,faces')
    %
    izw = netcdf.defVar(ncid,'mesh2d_zw','NC_DOUBLE',ifaces);
    netcdf.putAtt(ncid,izw,'standard_name','sea_surface_elevation')
    netcdf.putAtt(ncid,izw,'long_name','Water level')
    netcdf.putAtt(ncid,izw,'units','m')
    netcdf.putAtt(ncid,izw,'mesh','mesh2d')
    netcdf.putAtt(ncid,izw,'location','face')
    netcdf.putVar(ncid,izw,zw)
    %
    ih = netcdf.defVar(ncid,'mesh2d_h1','NC_DOUBLE',ifaces);
    netcdf.putAtt(ncid,ih,'standard_name','sea_floor_depth_below_sea_surface')
    netcdf.putAtt(ncid,ih,'long_name','Water depth')
    netcdf.putAtt(ncid,ih,'units','m')
    netcdf.putAtt(ncid,ih,'mesh','mesh2d')
    netcdf.putAtt(ncid,ih,'location','face')
    netcdf.putVar(ncid,ih,h)
    %
    izb = netcdf.defVar(ncid,'mesh2d_zb','NC_DOUBLE',inodes);
    netcdf.putAtt(ncid,izb,'standard_name','altitude')
    netcdf.putAtt(ncid,izb,'long_name','Bed level')
    netcdf.putAtt(ncid,izb,'units','m')
    netcdf.putAtt(ncid,izb,'mesh','mesh2d')
    netcdf.putAtt(ncid,izb,'location','node')
    netcdf.putVar(ncid,izb,zb)
    %
    iucx = netcdf.defVar(ncid,'mesh2d_ucx','NC_DOUBLE',ifaces);
    netcdf.putAtt(ncid,iucx,'standard_name','sea_water_x_velocity')
    netcdf.putAtt(ncid,iucx,'units','m s-1')
    netcdf.putAtt(ncid,iucx,'mesh','mesh2d')
    netcdf.putAtt(ncid,iucx,'location','face')
    netcdf.putVar(ncid,iucx,ucx)
    %
    iucy = netcdf.defVar(ncid,'mesh2d_ucy','NC_DOUBLE',ifaces);
    netcdf.putAtt(ncid,iucy,'standard_name','sea_water_y_velocity')
    netcdf.putAtt(ncid,iucy,'units','m s-1')
    netcdf.putAtt(ncid,iucy,'mesh','mesh2d')
    netcdf.putAtt(ncid,iucy,'location','face')
    netcdf.putVar(ncid,iucy,ucy)
catch Err
end
netcdf.close(ncid)
if ~isempty(Err)
    rethrow(Err)
end