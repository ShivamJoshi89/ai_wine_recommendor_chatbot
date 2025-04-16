// client/src/services/wineService.js
import api from './api';

/**
 * Retrieves a list of wines with optional filters.
 *
 * Supported filters:
 * @param {string} [filters.types]       Comma-separated wine types (e.g. "Red,Sparkling")
 * @param {number} [filters.min_price]   Minimum price
 * @param {number} [filters.max_price]   Maximum price
 * @param {number} [filters.min_rating]  Minimum Vivino average rating
 * @param {string} [filters.grapes]      Comma-separated grape types
 * @param {string} [filters.regions]     Comma-separated regions
 * @param {string} [filters.countries]   Comma-separated countries
 * @param {string} [filters.styles]      Comma-separated wine styles (primary_type)
 * @param {string} [filters.pairings]    Comma-separated food pairings
 * @param {number} [filters.limit=20]    Number of results per page
 * @param {number} [filters.page=1]      Page number (1-based)
 *
 * @returns {Promise<Array>} Array of wine objects.
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
 * @param {string} id The wine's unique identifier.
 * @returns {Promise<Object>} A wine object.
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
 * Retrieves the featured wines.
 * @returns {Promise<Array>} Array of featured wine objects.
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
 * @returns {Promise<Object>} Object with a "count" property.
 */
const getWinesCount = async (filters = {}) => {
  try {
    const response = await api.get('/wines/count', { params: filters });
    return response.data;
  } catch (error) {
    console.error('Error in getWinesCount:', error);
    throw error;
  }
};


/**
 * Performs a dynamic search for wines using MongoDB Atlas Search.
 * @param {string} searchQuery The search term entered by the user.
 * @param {number} [skip=0] Number of results to skip (for pagination).
 * @param {number} [limit=20] Maximum number of results to return.
 * @returns {Promise<Array>} Array of wine objects matching the search criteria.
 */
const getDynamicSearch = async (searchQuery, skip = 0, limit = 20) => {
  try {
    const response = await api.get('/wines/search', {
      params: { q: searchQuery, skip, limit },
    });
    return response.data;
  } catch (error) {
    console.error('Error in getDynamicSearch:', error);
    throw error;
  }
};

export { getWines, getWineDetail, getFeaturedWine, getWinesCount, getDynamicSearch };
