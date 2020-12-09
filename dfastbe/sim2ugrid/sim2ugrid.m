function sim2ugrid(filename)
%SIM2UGRID convert simulation results to netCDF UGRID file
%   The function copies only the data relevant for D-FAST Bank Erosion.

dateformat = 'yyyy-mm-ddTHH:MM:SS';
prehistory = '';
simOrg = qpfopen(filename);
ncfile = 'undefined';
switch simOrg.FileType
    case 'SIMONA SDS FILE'
        ncfile = [filename, '_map.nc'];
        [xd, yd] = waquaio(simOrg, '', 'dgrid');
        zb = waquaio(simOrg, '', 'height');
        %
        Info = waqua('read', simOrg, '', 'SOLUTION_FLOW_SEP', []);
        last_time = length(Info.SimTime);
        [zw, t] = waquaio(simOrg, '', 'wlvl', last_time);
        [h, t] = waquaio(simOrg, '', 'wdepth', last_time);
        [ucx, ucy, t] = waquaio(simOrg, '', 'xyveloc', last_time);
        [chu, chv] = waquaio(simOrg, '', 'chezy');
        czs = sqrt(4./(1./chu(:,[1 1:end-1]).^2 + 1./chu.^2 + 1./chv([1 1:end-1],:).^2 + 1./chv.^2));
        %
        node_active = ~isnan(xd(:));
        xnode = xd(node_active);
        ynode = yd(node_active);
        zb = zb(node_active);
        nnodes = length(xnode);
        %
        nodes = xd;
        nodes(node_active) = 1:nnodes;
        faces = cat(3, nodes([1 1:end-1], [1 1:end-1]), ...
                       nodes(:, [1 1:end-1]), ...
                       nodes, ...
                       nodes([1 1:end-1], :));
        face_active = all(~isnan(faces), 3) & ~isnan(zw);
        zw = zw(face_active);
        h = h(face_active);
        ucx = ucx(face_active);
        ucy = ucy(face_active);
        czs = czs(face_active);
        %
        faces = reshape(faces, numel(xd), 4);
        faces = faces(face_active, :);
        nfaces = size(faces, 1);
        %
        modelname = 'SIMONA';
        for i = length(simOrg.WriteProg):-1:1
            new_entry = [datestr(simOrg.WriteProg(i).Date, dateformat), ': ', simOrg.WriteProg(i).Name];
            if isempty(prehistory)
                prehistory = new_entry;
            else
                prehistory = [prehistory, '\n', new_entry];
            end
        end
    case 'NEFIS'
        switch simOrg.SubType
            case 'Delft3D-trim'
                [p,f,e] = fileparts(filname);
                ncfile = [p, filesep, f, '_map.nc'];
                xd = vs_get(simOrg, 'map-const', 'XCOR', 'quiet');
                yd = vs_get(simOrg, 'map-const', 'YCOR', 'quiet');
                node_active = xd~=0 & yd~=0;
                %
                Info = vs_disp(simOrg, 'map-series', []);
                last_time = Info.SizeDim;
                %
                [dps, success] = vs_get(simOrg, 'map-sed-series', {last_time}, 'DPS', 'quiet');
                if success
                    zb = -dps;
                    zb_loc = 'face';
                else
                    dp = vs_get(simOrg, 'map-const', 'DP0', 'quiet');
                    zb = -dp;
                    dpsopt = vs_get(simOrg, 'map-const', 'DRYFLP', 'quiet');
                    if isequal(lower(deblank(dpsopt)), 'dp')
                        zb_loc = 'face';
                    else
                        zb_loc = 'node';
                    end
                end
                %
                zw = vs_get(simOrg, 'map-series', {last_time}, 'S1', 'quiet');
                h = qpread(simOrg, 'water depth', 'data', last_time);
                h = h.Val;
                u = qpread(simOrg, 'depth averaged velocity', 'data', last_time);
                ucx = u.XComp;
                ucy = u.YComp;
                %
                error('CFUROU required. Please add Chezy = true to the mdf-file.')
                czs = sqrt(4./(1./chu(:,[1 1:end-1]).^2 + 1./chu.^2 + 1./chv([1 1:end-1],:).^2 + 1./chv.^2));
                %
                xnode = xd(node_active);
                ynode = yd(node_active);
                nnodes = length(xnode);
                %
                nodes = xd;
                nodes(node_active) = 1:nnodes;
                faces = cat(3, nodes([1 1:end-1], [1 1:end-1]), ...
                    nodes(:, [1 1:end-1]), ...
                    nodes, ...
                    nodes([1 1:end-1], :));
                face_active = all(~isnan(faces), 3) & ~isnan(zw);
                zw = zw(face_active);
                h = h(face_active);
                ucx = ucx(face_active);
                ucy = ucy(face_active);
                czs = czs(face_active);
                %
                switch zb_loc
                    case 'face'
                        zb = zb(face_active);
                    case 'node'
                        zb = zb(node_active);
                end
                %
                faces   = reshape(faces, numel(xd), 4);
                faces   = faces(face_active, :);
                nfaces  = size(faces, 1);
                %
                modelname = 'Delft3D';
                simdat = vs_get(simOrg, 'map-version', 'FLOW_SIMDAT', 'quiet');
                simdat = datenum(sscanf(simdat,'%4d%2d%2d %2d%2d%2d',[1 6]));
                prehistory = [datestr(simdat, dateformat), ': Delft3D-FLOW'];
            otherwise
                error('NEFIS %s files are not (yet) supported by SIM2UGRID.', simOrg.SubType)
        end
    otherwise
        error('%s files are not (yet) supported by SIM2UGRID.', simOrg.FileType)
end
%
ncid = netcdf.create(ncfile, 'NETCDF4');
Err = [];
try
    inodes = netcdf.defDim(ncid, 'mesh2d_nnodes', nnodes);
    ifaces = netcdf.defDim(ncid, 'mesh2d_nfaces', nfaces);
    imaxfn = netcdf.defDim(ncid, 'mesh2d_nmax_face_nodes', 4);
    %
    mesh = netcdf.defVar(ncid, 'mesh2d', 'NC_DOUBLE', []);
    netcdf.putAtt(ncid, mesh, 'cf_role', 'mesh_topology')
    netcdf.putAtt(ncid, mesh, 'topology_dimension', int32(2))
    netcdf.putAtt(ncid, mesh, 'node_coordinates', 'mesh2d_node_x mesh2d_node_y')
    netcdf.putAtt(ncid, mesh, 'face_node_connectivity', 'mesh2d_face_nodes')
    %
    ix = netcdf.defVar(ncid, 'mesh2d_node_x', 'NC_DOUBLE', inodes);
    netcdf.putAtt(ncid, ix, 'standard_name', 'projection_x_coordinate')
    netcdf.putAtt(ncid, ix, 'units', 'm')
    netcdf.putVar(ncid, ix, xnode)
    %
    iy = netcdf.defVar(ncid, 'mesh2d_node_y', 'NC_DOUBLE', inodes);
    netcdf.putAtt(ncid, iy, 'standard_name', 'projection_y_coordinate')
    netcdf.putAtt(ncid, iy, 'units', 'm')
    netcdf.putVar(ncid, iy, ynode)
    %
    ifnc = netcdf.defVar(ncid, 'mesh2d_face_nodes', 'NC_INT', [imaxfn, ifaces]);
    netcdf.putAtt(ncid, ifnc, 'cf_role', 'face_node_connectivity')
    netcdf.putAtt(ncid, ifnc, 'start_index', int32(1))
    netcdf.putVar(ncid, ifnc, faces')
    %
    izw = netcdf.defVar(ncid, 'mesh2d_zw', 'NC_DOUBLE', ifaces);
    netcdf.putAtt(ncid, izw, 'standard_name', 'sea_surface_elevation')
    netcdf.putAtt(ncid, izw, 'long_name', 'Water level')
    netcdf.putAtt(ncid, izw, 'units', 'm')
    netcdf.putAtt(ncid, izw, 'mesh', 'mesh2d')
    netcdf.putAtt(ncid, izw, 'location', 'face')
    netcdf.putVar(ncid, izw, zw)
    %
    ih = netcdf.defVar(ncid, 'mesh2d_h1', 'NC_DOUBLE', ifaces);
    netcdf.putAtt(ncid, ih, 'standard_name', 'sea_floor_depth_below_sea_surface')
    netcdf.putAtt(ncid, ih, 'long_name', 'Water depth')
    netcdf.putAtt(ncid, ih, 'units', 'm')
    netcdf.putAtt(ncid, ih, 'mesh', 'mesh2d')
    netcdf.putAtt(ncid, ih, 'location', 'face')
    netcdf.putVar(ncid, ih, h)
    %
    izb = netcdf.defVar(ncid, 'mesh2d_zb', 'NC_DOUBLE', inodes);
    netcdf.putAtt(ncid, izb, 'standard_name', 'altitude')
    netcdf.putAtt(ncid, izb, 'long_name', 'Bed level')
    netcdf.putAtt(ncid, izb, 'units', 'm')
    netcdf.putAtt(ncid, izb, 'mesh', 'mesh2d')
    netcdf.putAtt(ncid, izb, 'location', 'node')
    netcdf.putVar(ncid, izb, zb)
    %
    iucx = netcdf.defVar(ncid, 'mesh2d_ucx', 'NC_DOUBLE', ifaces);
    netcdf.putAtt(ncid, iucx, 'standard_name', 'sea_water_x_velocity')
    netcdf.putAtt(ncid, iucx, 'units', 'm s-1')
    netcdf.putAtt(ncid, iucx, 'mesh', 'mesh2d')
    netcdf.putAtt(ncid, iucx, 'location', 'face')
    netcdf.putVar(ncid, iucx, ucx)
    %
    iucy = netcdf.defVar(ncid, 'mesh2d_ucy', 'NC_DOUBLE', ifaces);
    netcdf.putAtt(ncid, iucy, 'standard_name', 'sea_water_y_velocity')
    netcdf.putAtt(ncid, iucy, 'units', 'm s-1')
    netcdf.putAtt(ncid, iucy, 'mesh', 'mesh2d')
    netcdf.putAtt(ncid, iucy, 'location', 'face')
    netcdf.putVar(ncid, iucy, ucy)
    %
    iczs = netcdf.defVar(ncid, 'mesh2d_czs', 'NC_DOUBLE', ifaces);
    netcdf.putAtt(ncid, iczs, 'long_name', 'Chezy roughness')
    netcdf.putAtt(ncid, iczs, 'units', 'm0.5s-1')
    netcdf.putAtt(ncid, iczs, 'mesh', 'mesh2d')
    netcdf.putAtt(ncid, iczs, 'location', 'face')
    netcdf.putVar(ncid, iczs, czs)
    %
    iglobal = netcdf.getConstant('GLOBAL');
    history = [sprintf('%s: sim2ugrid.m "%s"', datestr(now, dateformat), protect(filename)), prehistory];
    netcdf.putAtt(ncid, iglobal, 'history', history)
    netcdf.putAtt(ncid, iglobal, 'converted_from', modelname)
    netcdf.putAtt(ncid, iglobal, 'Conventions', 'CF-1.8 UGRID-1.0 Deltares-0.10')
catch Err
end
netcdf.close(ncid)
if ~isempty(Err)
    rethrow(Err)
end

function str = protect(str)
str = strrep(str, '\', '\\');