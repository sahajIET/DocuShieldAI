# Use this version if you know your app outputs to 'dist' directory
FROM --platform=linux/amd64 node:18-alpine AS build

WORKDIR /app

COPY package*.json ./
RUN npm ci --include=dev --legacy-peer-deps

COPY . .

ARG VITE_BACKEND_URL=http://localhost:8000
ENV VITE_BACKEND_URL=$VITE_BACKEND_URL

RUN npm run build

# Change this line based on your build output:
# For Vite: COPY --from=build /app/dist /usr/share/nginx/html
# For Create React App: COPY --from=build /app/build /usr/share/nginx/html

FROM nginx:stable-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]