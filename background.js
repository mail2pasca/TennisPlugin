let monitoringInterval = null;

chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
  if (request.action === 'startMonitoring') {
    startMonitoring(request.settings);
  } else if (request.action === 'stopMonitoring') {
    stopMonitoring();
  }
});

function startMonitoring(settings) {
  // Clear any existing monitoring
  stopMonitoring();

  // Create alarm for periodic checking
  chrome.alarms.create('checkPortal', {
    periodInMinutes: settings.checkInterval
  });

  // Store settings for use by alarm handler
  chrome.storage.local.set({ monitoringSettings: settings });

  // Show notification that monitoring has started
  chrome.notifications.create({
    type: 'basic',
    iconUrl: 'icon48.png',
    title: 'Tennis Court Booking Monitor',
    message: 'Monitoring has started. Will check every ' + settings.checkInterval + ' minutes.'
  });
}

function stopMonitoring() {
  // Clear the alarm
  chrome.alarms.clear('checkPortal');

  // Clear stored settings
  chrome.storage.local.remove('monitoringSettings');

  // Show notification that monitoring has stopped
  chrome.notifications.create({
    type: 'basic',
    iconUrl: 'icon48.png',
    title: 'Tennis Court Booking Monitor',
    message: 'Monitoring has been stopped.'
  });
}

// Handle alarm events
chrome.alarms.onAlarm.addListener(async (alarm) => {
  if (alarm.name === 'checkPortal') {
    const { monitoringSettings } = await chrome.storage.local.get('monitoringSettings');
    if (monitoringSettings) {
      checkPortalAndBook(monitoringSettings);
    }
  }
});

async function checkPortalAndBook(settings) {
  try {
    // Check if it's time to attempt booking
    if (isPortalOpenTime(settings.portalOpenTime)) {
      // Create a new tab with the booking URL
      const tab = await chrome.tabs.create({ url: settings.bookingUrl });

      // Wait for page to load
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Inject the content script to handle the booking process
      await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        function: performBooking,
        args: [settings]
      });
    }
  } catch (error) {
    console.error('Booking attempt failed:', error);
    notifyError(error);
  }
}

function isPortalOpenTime(portalOpenTime) {
  const now = new Date();
  const [hours, minutes] = portalOpenTime.split(':');
  const portalTime = new Date();
  portalTime.setHours(parseInt(hours), parseInt(minutes), 0);

  // Check if current time is within 2 minutes of portal opening time
  const timeDiff = Math.abs(now - portalTime);
  return timeDiff <= 2 * 60 * 1000; // 2 minutes in milliseconds
}

function notifyError(error) {
  chrome.notifications.create({
    type: 'basic',
    iconUrl: 'icon48.png',
    title: 'Booking Error',
    message: 'Failed to book court: ' + error.message
  });
}

// Content script function to be injected
function performBooking(settings) {
  async function login() {
    // Find login form fields (modify selectors based on the actual website)
    const usernameField = document.querySelector('input[type="text"]');
    const passwordField = document.querySelector('input[type="password"]');
    const loginButton = document.querySelector('button[type="submit"]');

    if (usernameField && passwordField && loginButton) {
      // Fill in credentials
      usernameField.value = settings.username;
      passwordField.value = settings.password;

      // Trigger events
      usernameField.dispatchEvent(new Event('input', { bubbles: true }));
      passwordField.dispatchEvent(new Event('input', { bubbles: true }));

      // Click login button
      loginButton.click();
      return true;
    }
    return false;
  }

  async function selectDateTime() {
    // Wait for booking page to load
    await new Promise(resolve => setTimeout(resolve, 2000));

    // Find date/time selection elements (modify selectors based on the actual website)
    const dateField = document.querySelector('input[type="date"]');
    const timeField = document.querySelector('input[type="time"]');
    const bookButton = document.querySelector('button[data-action="book"]');

    if (dateField && timeField && bookButton) {
      // Set date and time
      dateField.value = settings.date;
      timeField.value = settings.time;

      // Trigger events
      dateField.dispatchEvent(new Event('input', { bubbles: true }));
      timeField.dispatchEvent(new Event('input', { bubbles: true }));

      // Click book button
      bookButton.click();
      return true;
    }
    return false;
  }

  // Execute booking process
  (async () => {
    try {
      // Check if portal is open (look for specific elements or messages)
      const portalClosed = document.querySelector('.portal-closed-message');
      if (portalClosed) {
        throw new Error('Portal is not open yet');
      }

      // Attempt login
      const loginSuccess = await login();
      if (!loginSuccess) {
        throw new Error('Login failed - could not find form elements');
      }

      // Wait for login to complete
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Attempt to select date/time and book
      const bookingSuccess = await selectDateTime();
      if (!bookingSuccess) {
        throw new Error('Booking failed - could not find booking elements');
      }

      // Notify success
      chrome.runtime.sendMessage({
        action: 'bookingSuccess',
        message: 'Court booked successfully!'
      });

    } catch (error) {
      chrome.runtime.sendMessage({
        action: 'bookingError',
        message: error.message
      });
    }
  })();
}