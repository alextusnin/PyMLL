clear

% Apply the new default colors to the current axes.
newDefaultColors = lines(6);
set(gca, 'ColorOrder', newDefaultColors, 'NextPlot', 'replacechildren');
newColorOrder = get(gca,'ColorOrder');

data_gain = load('fcutline_Archi_21cm2_2um1.mat');
sel_idx = [-9]; % select input signal power in dBm [..., -9, -10, -13, ...]
coupling_loss_1550 = 5.8; % coupling loss in dB, see insertion loss characterization file
signal_loss = coupling_loss_1550/2; % dB

s1 = subplot(1,1,1)
for idx_seq = 1:length(sel_idx)
    idx = find(data_gain.trs.signal_power == sel_idx(idx_seq))
    [filtered_x, filtered_y] = remove_outlier(data_gain.trs.wl_gain{(idx)}(1,2:end), data_gain.trs.wl_gain{(idx)}(2,2:end), 10);

    plot(data_gain.trs.wl_gain{(idx)}(1,2:end), movmean(data_gain.trs.wl_gain{(idx)}(2,2:end),4)+coupling_loss_1550, '-','LineWidth',1, 'color', newDefaultColors(idx_seq,:),...
        'MarkerFaceColor',newDefaultColors(idx_seq,:), 'MarkerEdgeColor',newDefaultColors(idx_seq,:),...
        'DisplayName', strcat(num2str(round(10^((data_gain.trs.signal_power(idx)-signal_loss)/10),2)),'mW'))
    hold on
end
lgd = legend('Location','Best');
lgd.FontSize = 12;
legend boxoff;

for idx_seq = 1:length(sel_idx)
    idx = find(data_gain.trs.signal_power == sel_idx(idx_seq))
    [filtered_x, filtered_y] = remove_outlier(data_gain.trs.wl_gain{(idx)}(1,2:end), data_gain.trs.wl_gain{(idx)}(2,2:end)+coupling_loss_1550, 10);
    scatter(filtered_x, filtered_y, 'o',...
        'MarkerFaceColor', newDefaultColors(idx_seq,:),'MarkerEdgeColor',newDefaultColors(idx_seq,:))
    hold on

end

set(gca,'LineWidth',1)
box on

fontsize = 14;
xlabel('Wavelength (nm)', 'Fontsize', fontsize);
ylabel('On-chip Net Gain (dB)', 'Fontsize', fontsize);
s1.XAxis.FontSize = fontsize;
s1.YAxis.FontSize = fontsize;
xlim([1530 1580])
ylim([15, 35]);

set(gcf,'units','points','position',[10 10 800 300])
%savefig(gcf, 'Figure_3B_broadband_gain.fig');

% remove outliers that might be caused by parasitic lasing
length = 0.21 %meters
gain_per_m =  filtered_y/10/log10(exp(1))/length
function [filtered_x, filtered_y] = remove_outlier(x_data, y_data, win_size)
    [B, TF] = rmoutliers(y_data,'movmedian',win_size);
    filtered_x = x_data((~TF));
    filtered_y = B;    
end

