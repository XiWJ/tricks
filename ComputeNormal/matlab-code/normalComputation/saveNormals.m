function saveNormals(target,normal_vector, depthValid, type)
    %packaging everything in a parfor is a pain
    if type == '.npy'
        writeNPY(normal_vector, target);
        target_mask = strrep(target, 'normal', 'normal_mask');
        writeNPY(depthValid, target_mask);
    else
        save(target,'normal_vector', 'depthValid');
    end
end
