infolderName = '2017-10-ZhenjiangBallast';
fnames = getAllFilesInFolder(infolderName);
parsed = get_cell_tokens(fnames, '-');

outfolderName = 'EUIAIA_convertion';
if ~exist(outfolderName, 'dir')
	mkdir(outfolderName);
end

% in Wei's file, the nomenclature is 'x(No.)' + '-' + 'front(0)/top(1)/side(2)' + '-'+ 'seg.bmp' for
% binary images, 'x(No.)-front/side/top.bmp' for original images. Therefore
% length of 3 means the binary images.
% in EUIAIA, the nomenclature is 'timg' + '0(front)/1(top)/2(side)' + '000x(No.).png'
for i = 1:length(parsed)
    if length(parsed{i}) == 3
        x = str2num(parsed{1,i}{1}); % the first element is x(No.), store as a number
        viewFrom = parsed{1,i}{2}; % the second element is front(0)/top(1)/side(2), store as a string
        
        im = imread(fullfile(infolderName,fnames{1,i}));
        switch viewFrom
            case 'front'
                newName = strcat('timg', '0',sprintf('%04d', x),'.png');
            case 'top'
                newName = strcat('timg', '1',sprintf('%04d', x),'.png');
            case 'side'
                newName = strcat('timg', '2',sprintf('%04d', x),'.png');
        end
        imwrite(im,fullfile(outfolderName,newName));
        
    end
end
