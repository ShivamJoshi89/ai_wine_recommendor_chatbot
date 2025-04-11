// client/src/services/wineService.js
import api from './api';

/**
 * Retrieves a list of wines with optional filters.
 */
const getWines = async (filters = {}) => {
  try {
    const response = await api.get('/wines', { params: filters });
    return response.data;
  } catch (error) {
    throw error;
  }
};

/**
 * Retrieves detailed information for a specific wine.
 */
const getWineDetail = async (id) => {
  try {
    const response = await api.get(`/wines/${id}`);
    return response.data;
  } catch (error) {
    throw error;
  }
};

/**
 * Retrieves the featured wine (the wine with the highest rating).
 */
const getFeaturedWine = async () => {
  try {
    const response = await api.get('/wines/featured');
    return response.data;
  } catch (error) {
    throw error;
  }
};

/**
 * Retrieves the total count of wines.
 * Make sure your backend has an endpoint /wines/count that returns an object like { count: number }
 */
const getWinesCount = async () => {
  try {
    const response = await api.get('/wines/count');
    return response.data;
  } catch (error) {
    throw error;
  }
};

export { getWines, getWineDetail, getFeaturedWine, getWinesCount };
