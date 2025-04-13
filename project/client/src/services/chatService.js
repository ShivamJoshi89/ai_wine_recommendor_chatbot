// client/src/services/chatService.js
import api from './api';

/**
 * Sends a chat message and returns the assistant response.
 * @param {string} message - The user's chat message.
 * @param {string} token - The JWT access token for authentication.
 * @returns {Promise<Object>} - An object containing the assistant's response.
 */
const sendMessage = async (message, token) => {
  try {
    const response = await api.post(
      '/chat',
      { message },
      {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      }
    );
    return response.data;
  } catch (error) {
    console.error('Error in sendMessage:', error);
    throw error;
  }
};

export { sendMessage };
