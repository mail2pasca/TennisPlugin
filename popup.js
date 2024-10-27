document.addEventListener('DOMContentLoaded', function() {
    // Load saved settings and monitoring status
    chrome.storage.sync.get([
      'bookingUrl',
      'username',
      'date',
      'time',
      'portalOpenTime',
      'checkInterval',
      'isMonitoring'
    ], function(result) {
      document.getElementById('bookingUrl').value = result.bookingUrl || '';
      document.getElementById('username').value = result.username || '';
      document.getElementById('date').value = result.date || '';
      document.getElementById('time').value = result.time || '';
      document.getElementById('portalOpenTime').value = result.portalOpenTime || '';
      document.getElementById('checkInterval').value = result.checkInterval || '60';
      
      updateMonitoringStatus(result.isMonitoring);
    });
  
    // Save settings
    document.getElementById('saveSettings').addEventListener('click', function() {
      const settings = {
        bookingUrl: document.getElementById('bookingUrl').value,
        username: document.getElementById('username').value,
        date: document.getElementById('date').value,
        time: document.getElementById('time').value,
        portalOpenTime: document.getElementById('portalOpenTime').value,
        checkInterval: document.getElementById('checkInterval').value
      };
  
      chrome.storage.sync.set(settings, function() {
        updateStatus('Settings saved successfully!');
      });
    });
  
    // Start monitoring
    document.getElementById('startMonitoring').addEventListener('click', function() {
      const settings = {
        bookingUrl: document.getElementById('bookingUrl').value,
        username: document.getElementById('username').value,
        password: document.getElementById('password').value,
        date: document.getElementById('date').value,
        time: document.getElementById('time').value,
        portalOpenTime: document.getElementById('portalOpenTime').value,
        checkInterval: parseInt(document.getElementById('checkInterval').value)
      };
  
      chrome.runtime.sendMessage({
        action: 'startMonitoring',
        settings: settings
      });
  
      chrome.storage.sync.set({ isMonitoring: true });
      updateMonitoringStatus(true);
    });
  
    // Stop monitoring
    document.getElementById('stopMonitoring').addEventListener('click', function() {
      chrome.runtime.sendMessage({ action: 'stopMonitoring' });
      chrome.storage.sync.set({ isMonitoring: false });
      updateMonitoringStatus(false);
    });
  
    function updateStatus(message) {
      const statusDiv = document.getElementById('status');
      statusDiv.textContent = message;
      setTimeout(() => {
        statusDiv.textContent = '';
      }, 3000);
    }
  
    function updateMonitoringStatus(isMonitoring) {
      const startButton = document.getElementById('startMonitoring');
      const stopButton = document.getElementById('stopMonitoring');
      const statusDiv = document.getElementById('status');
  
      if (isMonitoring) {
        startButton.style.display = 'none';
        stopButton.style.display = 'block';
        statusDiv.textContent = 'Monitoring active';
        statusDiv.classList.add('monitoring-active');
      } else {
        startButton.style.display = 'block';
        stopButton.style.display = 'none';
        statusDiv.textContent = 'Monitoring inactive';
        statusDiv.classList.remove('monitoring-active');
      }
    }
  });  