// client/src/services/authService.js
import api from './api';

/**
 * Registers a new user.
 * @param {Object} userData - An object containing username, email, and password.
 * @returns {Promise<Object>} - The registered user data.
 */
const register = async (userData) => {
  try {
    const response = await api.post('/auth/register', userData);
    return response.data;
  } catch (error) {
    throw error;
  }
};

/**
 * Logs in a user.
 * @param {Object} credentials - An object containing username and password.
 * @returns {Promise<Object>} - The response with access token and token type.
 */
const login = async (credentials) => {
  try {
    const params = new URLSearchParams();
    params.append("username", credentials.username);
    params.append("password", credentials.password);
    const response = await api.post('/auth/login', params, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded'
      }
    });
    return response.data;
  } catch (error) {
    throw error;
  }
};

export { register, login };
