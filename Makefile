init:
	docker compose up -d --build

stop:        
	docker compose down

reset:        
	docker compose down -v

logs-web:
	docker compose logs -f airflow-web
