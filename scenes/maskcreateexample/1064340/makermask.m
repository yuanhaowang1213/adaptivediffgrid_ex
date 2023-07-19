clear all;
clc;
filename_dir = './';
% ori_image_file = fullfile(filename_dir, '10643_40_bin5.mrc');
fidlocfile = fullfile(filename_dir, 'b2tilt40_bin5_ali.fid.txt');
fidloc = load(fidlocfile);
isize = [820,820,41];
fidlocs = cell(isize(3),1);
for i = 1 : isize(3)
    fidlocs{i} = fidloc(fidloc(:,5)==(i-1),3:4);
end
radi = 9; %%9 for 10643
radira2 = 1.2*radi;
ceilradi2 = ceil(radira2);
[maskidy2, maskidx2] = meshgrid(-ceilradi2 : 1 : ceilradi2, -ceilradi2 : 1 : ceilradi2);
maskid2 = zeros(size(maskidy2));
maskid2_size = size(maskid2);
for i  = 1 : maskid2_size(1)
    for j = 1: maskid2_size(2)
        if(maskidx2(i,j)^2 + maskidy2(i,j)^2 >= radi^2) && (maskidx2(i,j)^2 + maskidy2(i,j)^2 < radira2^2)
            maskid2(i,j) = 1;
        end
    end
end

mask2 = ones(isize);

[imagey, imagex] = meshgrid(1:isize(2), 1 : isize(1));

maskmarkerdir = fullfile(filename_dir,'maskmarker');
mkdir(maskmarkerdir);
%% around mean use this code
for slicenum = 1 : isize(3)
%     slicenum
    masktmp = true(isize(1),isize(2));
    for k = 1 : size(fidlocs{slicenum},1)
        mask= (imagey-fidlocs{slicenum}(k,2)).^2 + (imagex-fidlocs{slicenum}(k,1)).^2 < radi^2;
        masktmp(mask) = false;
%         end
    end
    mask2(:,:,slicenum) = masktmp;
%     datatmp = data1(:,:,slicenum);
%     datatmp(not(masktmp)) = mean(datatmp(masktmp));
%     data1(:,:,slicenum) = datatmp;
    maskname = sprintf('%04d.png',slicenum-1);
    imwrite(single(masktmp), fullfile(maskmarkerdir, maskname),'PNG');
end
% writeanalyze(single(mask2(:,:,:)),size(mask2(:,:,:)), fullfile(filename_dir,'mask.hdr'), [1.,1.,1.])