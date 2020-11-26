sgtitle('Step response 22.1163°C to 30°C')

subplot(2,1,1)
plot(out.t, out.Reference)
hold on
plot(out.t, out.Temperature)

title('Step Response')
xlabel('time (t)')
ylabel('Temperature(°C)')
legend({'Reference (°C)','Step Response (°C)'})

subplot(2,1,2)

plot(out.t, out.Error)

title('Error')
xlabel('time (t)')
ylabel('Error(°C)')
legend({'Error Signal'})