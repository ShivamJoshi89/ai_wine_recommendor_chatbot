// client/src/services/wineService.js
import api from './api';

/**
 * Retrieves a list of wines with optional filters.
 * @param {object} filters - Query parameters for wine filtering.
 * @returns {Promise<Array>} - Array of wine objects.
 */
const getWines = async (filters = {}) => {
  try {
    const response = await api.get('/wines', { params: filters });
    return response.data;
  } catch (error) {
    console.error('Error in getWines:', error);
    throw error;
  }
};

/**
 * Retrieves detailed information for a specific wine.
 * @param {string} id - The wine's unique identifier.
 * @returns {Promise<Object>} - A wine object.
 */
const getWineDetail = async (id) => {
  try {
    const response = await api.get(`/wines/${id}`);
    return response.data;
  } catch (error) {
    console.error('Error in getWineDetail:', error);
    throw error;
  }
};

/**
 * Retrieves the featured wine(s) according to our business logic.
 * @returns {Promise<Array>} - Array of featured wine objects.
 */
const getFeaturedWine = async () => {
  try {
    const response = await api.get('/wines/featured');
    return response.data;
  } catch (error) {
    console.error('Error in getFeaturedWine:', error);
    throw error;
  }
};

/**
 * Retrieves the total count of wines.
 * @returns {Promise<Object>} - Object with a "count" property.
 */
const getWinesCount = async () => {
  try {
    const response = await api.get('/wines/count');
    return response.data;
  } catch (error) {
    console.error('Error in getWinesCount:', error);
    throw error;
  }
};

export { getWines, getWineDetail, getFeaturedWine, getWinesCount };
